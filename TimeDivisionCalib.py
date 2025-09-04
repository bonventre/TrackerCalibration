import sys
import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.stats

import dbUtility

halflengths = np.array([587.878, 585.846, 583.79, 581.711, 579.607, 577.478, 575.325, 573.146, 570.942, 568.713, 566.457, 564.175, 561.866, 559.531, 557.168, 554.777, 552.359, 549.912, 547.436, 544.931, 542.396, 539.832, 537.237, 534.611, 531.954, 529.265, 526.544, 523.79, 521.003, 518.182, 515.326, 512.436, 509.51, 506.547, 503.548, 500.512, 497.437, 494.324, 491.17, 487.977, 484.742, 481.465, 478.146, 474.782, 471.374, 467.921, 464.42, 460.872, 457.275, 453.627, 449.929, 446.178, 442.373, 438.513, 434.596, 430.62, 426.585, 422.488, 418.328, 414.102, 409.809, 405.447, 401.012, 396.504, 391.918, 387.253, 382.506, 377.673, 372.752, 367.737, 362.627, 357.416, 352.101, 346.676, 341.135, 335.475, 329.687, 323.766, 317.704, 311.493, 305.123, 298.585, 291.866, 284.955, 277.838, 270.496, 262.913, 255.067, 246.931, 238.477, 229.67, 220.467, 210.817, 200.654, 189.897, 178.438])
lengths = halflengths*2

def readData(filenames):
    hitfields = ["plane","panel","straw","uupos","etime","ulresid","ulresidpvar","ulresidmvar","state","edep","wdist","udt","udoca"]
    trkfields = ["npanels"]
    run = 0
    a = {field: [] for field in hitfields+trkfields}
    for fn in filenames:
        f = uproot.open(fn)
        t = f["EventNtuple/ntuple"].arrays(filter_name=["trkhits","trk/*","evtinfo/*"])
        for field in hitfields:
            a[field].append(np.copy(ak.flatten(ak.flatten(t["trkhits"][field])).to_numpy()))
        for field in trkfields:
            a[field].append(np.copy(ak.flatten(ak.flatten(ak.broadcast_arrays(t["trkhits"][hitfields[0]],t["trk.%s" % field])[1])).to_numpy()))
        run = int(t["run"][0])
    for field in a:
        a[field] = np.concatenate(a[field])

    a["sid"] = a["plane"]*6*96 + a["panel"]*96 + a["straw"]
    a["deltat"] = a["etime"][:,1]-a["etime"][:,0]
    a["strawlen"] = halflengths[a["straw"]]
    # order by strawid so we can efficiently loop over channels
    order = ak.argsort(a["sid"])
    for field in a:
        a[field] = a[field][order]
    return a,run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run calibration')
    parser.add_argument('--output','-o',help='output filename stub (<output>.fcl, <output>.txt)',default='output')
    parser.add_argument('--dbpurpose','-p',help='database purpose',default='')
    parser.add_argument('--dbversion','-v',help='database version',default='')
    parser.add_argument('--dbtext','-t',help='comma separated list of database text files',default='')
    parser.add_argument('--source','-s',help='input SHD filename',default='')
    parser.add_argument('--source-list','-S',help='list of input SHD filenames',default='')
    parser.add_argument('--mincount','-c',type=int,help='minimum hit count',default=100)
    parser.add_argument('--dtcut','-d',type=float,help='dt cut (ns)',default=30.0)
    parser.add_argument('--strawfrac','-f',type=float,help='straw length fraction to fit',default=0.75)
    parser.add_argument('--ebins','-e',type=int,help='number of energy bins',default=12)
    parser.add_argument('--ewidth','-w',type=float,help='energy bin width (KeV)',default=0.5)
    parser.add_argument('--plot',action='store_true',help='plot result')

    args = parser.parse_args()
    if args.dbtext:
        args.dbtext = args.dbtext.split(",")
    else:
        args.dbtext = []

    if args.source:
        fns = [args.source]
    elif args.source_list:
        f = open(args.source_list)
        fns = [line.strip() for line in f.readlines()]
    else:
        print("Error: need to provide source or source list")
        sys.exit(1)
 
    a, run = readData(fns)
    print("Read data")
    
    c = dbUtility.getCalibrations(run,["TrkPreampStraw","TrkDelayPanel"],args.dbpurpose,args.dbversion,args.dbtext)
    print("Read calibrations")
    
    propvels = np.zeros(36*6*96)
    t0offsets = np.zeros(36*6*96)

    # calculate a correction for straws that may be way out
    # looks like this happens for some channels with high noise
    counts = ak.run_lengths(a["sid"])
    goodstraw = (counts >= args.mincount)
    mean_wdist = np.mean(ak.unflatten(a["wdist"],counts),axis=1)
    sids = a["sid"][np.concatenate(([0], np.cumsum(counts[:-1])))]
    wdist_straws = 0
    for i in range(len(goodstraw)):
        if goodstraw[i] and np.abs(mean_wdist[i]) > 800:
            t0offsets[sids[i]] = mean_wdist[i]/800.
            wdist_straws += 1
            print("setting default for",i,sids[i],t0offsets[sids[i]],np.mean(a["deltat"][a["sid"] == sids[i]]))
    print("%d straws had large wdist" % (wdist_straws))

    a["deltat"] -= t0offsets[a["sid"]]
    cut = (np.abs(a["deltat"]) < args.dtcut) & (a["state"] > -2) & (np.sqrt(a["ulresidpvar"]) < 50) & (np.abs(a["uupos"]) < a["strawlen"]*args.strawfrac) & (np.abs(a["ulresid"]) < 1000)

    for field in a:
        a[field] = a[field][cut]

    counts = ak.run_lengths(a["sid"])
    chan_starts = np.cumsum(np.concatenate(([0], counts[:-1])))
    chan_ends = np.cumsum(counts)
    sids = a["sid"][np.concatenate(([0], np.cumsum(counts[:-1])))]

    print("Using %d hits" % (np.sum(counts)))

    t0errs = []
    
    for i in range(len(sids)):
        sid = sids[i]
        if counts[i] < args.mincount:
            continue
        (slope, intercept), cov = np.polyfit(a["uupos"][chan_starts[i]:chan_ends[i]],a["deltat"][chan_starts[i]:chan_ends[i]],1,cov=True)
        t0errs.append(np.sqrt(cov[1][1]))
        propvels[sid] = 1/slope
        t0offsets[sid] = intercept

    #print("MEAN T0ERR: %.2f %.2f %.2f %.2f(ns)" % (np.mean(t0errs),np.std(t0errs),np.max(t0errs),np.min(t0errs)))
    print("MEAN T0ERR: %.2f (ns)" % (np.mean(t0errs)))
    print("Updated %d channel t0 offsets, avg mag = %.2f (ns)" % (np.sum(counts >= args.mincount),np.mean(np.abs(t0offsets[t0offsets != 0]))))

    energies = np.zeros(12)
    escales = np.zeros(12)
    
    pred = (a["deltat"] - t0offsets[a["sid"]])*propvels[a["sid"]]
    
    for i in range(args.ebins):
        emid = i*args.ewidth/1000.
        ecut = (np.abs(a["edep"]-emid) < args.ewidth/2./1000.) & (propvels[a["sid"]] != 0)
        uupose = a["uupos"][ecut]
        prede = pred[ecut]
    
        slope, intercept = np.polyfit(uupose,prede,1)
        energies[i] = emid
        escales[i] = 1/slope
    pval = propvels[a["sid"]] * np.interp(a["edep"], energies, escales)
    pred2 = (a["deltat"] - t0offsets[a["sid"]])*pval

    residuals = []
    residualerrs = []

    emids = []
    eslopes = []
    for j in range(args.ebins):
        uposes = []
        errors = []
        emid = j*args.ewidth/1000.
        ecut = (np.abs(a["edep"]-emid) < args.ewidth/2./1000.)
        for i in range(17):
            uuposmid = (i-8)*400/8.
            cut = (np.abs(a["uupos"] - uuposmid) < 25) & (pval != 0) & (ecut)
            fullvar = np.std(a["uupos"][cut]-pred2[cut])**2
            fitvar = np.mean(a["ulresidpvar"][cut])
            if fullvar > fitvar:
                measerr = np.sqrt(fullvar-fitvar)
            else:
                measerr = 0
            uposes.append(uuposmid)
            errors.append(measerr)
        uposes2 = [0]
        errors2 = [errors[8]]
        for i in range(1,9):
            uposes2.append(uposes[8+i])
            errors2.append((errors[8+i]+errors[8-i])/2.)
        slope, intercept = np.polyfit(uposes2,errors2,1)
        emids.append(intercept)
        eslopes.append(slope)
        if args.plot:
            plt.plot(uposes,errors,label="%d" % j)
    if args.plot:
        plt.show()
        input()

    goodcalib = (propvels[a["sid"]] != 0)
    startresids = (a["uupos"]-a["wdist"])[goodcalib]
    startresiderrs = np.sqrt(a["ulresidmvar"]+a["ulresidpvar"])[goodcalib]
    resids = (a["uupos"] - pred2)[goodcalib]
    residerrs = (np.interp(a["edep"],energies,emids) + np.interp(a["edep"],energies,eslopes)*np.abs(pred2))[goodcalib]

    startcore = np.abs(startresids) < 250
    upcore = np.abs(resids) < 250

    print("Starting calibration: (all, core < 250 mm)")
    mu, sigma = scipy.stats.norm.fit(startresids[startcore])
    print("Mean residual: %6.1f %6.1f" % (np.mean(startresids),mu))
    print("residual RMS : %6.1f %6.1f" % (np.std(startresids),sigma))
    pull = startresids/startresiderrs
    mu, sigma = scipy.stats.norm.fit(pull[np.abs(pull) < 2])
    print("pull RMS     : %6.2f %6.2f" % (np.std(pull),sigma))
    print("Updated calibration:")
    mu, sigma = scipy.stats.norm.fit(resids[upcore])
    print("Mean residual: %6.1f %6.1f" % (np.mean(resids),mu))
    print("residual RMS : %6.1f %6.1f"  % (np.std(resids),sigma))
    pull = resids/residerrs
    mu, sigma = scipy.stats.norm.fit(pull[np.abs(pull) < 2])
    print("pull RMS     : %6.2f %6.2f"  % (np.std(pull),sigma))
    import pdb;pdb.set_trace()

    fout = open(args.output + ".fcl","w")
    fout.write("services.ProditionsService.strawResponse.eBins : %d\n" % args.ebins)
    fout.write("services.ProditionsService.strawResponse.eBinWidth : %f\n" % args.ewidth)
    fout.write("services.ProditionsService.strawResponse.eHalfPVScale : [")
    for i in range(12):
        fout.write("%.2f" % escales[i])
        if i < 12-1:
            fout.write(", ")
    fout.write("]\n")
    fout.write("services.ProditionsService.strawResponse.defaultHalfPropVelocity : 100.0\n")
    fout.write("services.ProditionsService.strawResponse.centralWirePos : 0.0\n")
    fout.write("services.ProditionsService.strawResponse.tdCentralRes : [")
    for i in range(12):
        fout.write("%.2f" % emids[i])
        if i < 12-1:
            fout.write(", ")
    fout.write("]\n")
    fout.write("services.ProditionsService.strawResponse.tdResSlope : [")
    for i in range(12):
        fout.write("%.2f" % eslopes[i])
        if i < 12-1:
            fout.write(", ")
    fout.write("]\n")
    fout.write("services.ProditionsService.strawResponse.strawHalfPropVelocity : [\n")
    #for i in range(36):
    #    for j in range(6):
    #        for k in range(96):
    # calculate the mean propagation velocity for this straw # across all panels
    for k in range(96):
        mv = 0
        mc = 0
        for i in range(36):
            for j in range(6):
                s = i*6*96+j*96+k
                if propvels[s] != 0.0:
                    mv += propvels[s]
                    mc += 1
        if mc > 0:
            mv /= mc
            fout.write("%.2f" % mv)
        else:
            fout.write("100.0")
        if k != 95:
            fout.write(", ")
    fout.write("]\n")
    fout.close()

    for i in range(36*6*96):
        c["TrkPreampStraw"]["delay_cal"][i] = "%f" % (float(c["TrkPreampStraw"]["delay_cal"][i])+t0offsets[i]/2.)
        c["TrkPreampStraw"]["delay_hv"][i] = "%f" % (float(c["TrkPreampStraw"]["delay_hv"][i])-t0offsets[i]/2.)
    #FIXME panelmeanoffsets
    dbUtility.writeCalibrations(c,["TrkPreampStraw","TrkDelayPanel"],args.output+".txt")
    print("Wrote calibrations")

    if args.plot:
        import ROOT
#        hr0 = ROOT.TH1F("hr0","hr0",600,-600,600)
#        hr1 = ROOT.TH1F("hr1","hr1",600,-600,600)
#        hr0.FillN(len(a["ulresid"][goodcalib]),a["ulresid"][goodcalib].astype(np.double),np.ones(len(a["ulresid"][goodcalib]),dtype=np.double))
#        hr1.FillN(len(resids),resids.astype(np.double),np.ones(len(resids),dtype=np.double))
        hr0 = ROOT.TH1F("hr0","hr0",600,-6,6)
        hr1 = ROOT.TH1F("hr1","hr1",600,-6,6)
        hr0.FillN(len(startresids),(startresids/startresiderrs).astype(np.double),np.ones(len(startresids),dtype=np.double))
        hr1.FillN(len(resids),(resids/residerrs).astype(np.double),np.ones(len(resids),dtype=np.double))
        hr0.Draw()
        hr1.SetLineColor(ROOT.kRed)
        hr1.Draw("same")

        #plt.legend()
        #plt.show()
        input()
