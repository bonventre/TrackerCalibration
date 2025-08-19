import awkward as ak
import uproot
import sys
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run calibration')
    parser.add_argument('--output','-o',help='output filename stub (<output>.fcl, <output>.txt)',default='output')
    parser.add_argument('--source','-s',help='input SHD filename',default='')
    parser.add_argument('--source-list','-S',help='list of input SHD filenames',default='')
    parser.add_argument('--usemc',action='store_true',help='MC truth calibration')
    parser.add_argument('--pcode','-p',type=int,help='process code for MC',default=167)
    parser.add_argument('--ebins','-e',type=int,help='number of energy bins',default=12)
    parser.add_argument('--ewidth','-w',type=float,help='energy bin width (MeV)',default=0.0005)
    parser.add_argument('--tbins','-t',type=int,help='number of tot bins',default=16)
    parser.add_argument('--twidth','-W',type=float,help='tot bin width',default=5)
    parser.add_argument('--mint','-m',type=float,help='minimum udt (ns)',default=-10)
    parser.add_argument('--maxt','-M',type=float,help='maximum udt (ns)',default=60)
    parser.add_argument('--mincount','-c',type=int,help='minimum hit count',default=10)
    parser.add_argument('--plot',action='store_true',help='plot result')

    args = parser.parse_args()

    if args.source:
        fns = [args.source]
    elif args.source_list:
        f = open(args.source_list)
        fns = [line.strip() for line in f.readlines()]
    else:
        print("Error: need to provide source or source list")
        sys.exit(1)
 
    fields = ["tot","etime","edep","state","udt","utresid","utresidmvar"]
    mcfields = ["startCode","earlyend","t0","tprop"]
    a = {field : [] for field in fields}
    if args.usemc:
        for field in mcfields:
            a[field] = []
    for batch in uproot.iterate([fn + ":EventNtuple/ntuple" for fn in fns],filter_name=["trkhitsmc","trkhits","trk.fitcon","trk.nactive","trk.nhits"]):
        for field in fields:
            a[field].append(np.copy(ak.flatten(ak.flatten(batch["trkhits"][field])).to_numpy()))
        if args.usemc:
            for field in mcfields:
                a[field].append(np.copy(ak.flatten(ak.flatten(batch["trkhitsmc"][field][ak.local_index(batch["trkhitsmc"][field]) < ak.num(batch["trkhits"][fields[0]],axis=2)])).to_numpy()))
    
    for field in a:
        a[field] = np.concatenate(a[field])
    
    if args.usemc:
        cut = (a["startCode"] == args.pcode) & (a["state"] > -2)
    else:
        cut = (a["state"] > -2) & (a["udt"] > args.mint) & (a["udt"] < args.maxt)
    for field in a:
        a[field] = a[field][cut]
    if not args.usemc:
        a["earlyend"] = np.argmin(a["etime"],axis=1)

    a["earlytime"] = np.where(a["earlyend"] == 0,a["etime"][:,0],a["etime"][:,1])
    a["earlytot"] = np.where(a["earlyend"] == 0,a["tot"][:,0],a["tot"][:,1])
    if args.usemc:
        a["tdrift"] = a["earlytime"] - a["t0"] - a["tprop"]
    else:
        a["tdrift"] = a["udt"]

    default_drift = np.mean(a["tdrift"])
    default_std = np.std(a["tdrift"])
    means = np.zeros((args.tbins,args.ebins))
    stds = np.zeros((args.tbins,args.ebins))
    
    resids = []
    residerrs = []
    ibins = []
    jbins = []

    print("Using %d hits for calibration" % (len(a["tdrift"])))
   
    for i in range(args.tbins):
        for j in range(args.ebins):
            tmin = args.twidth*(i-0.5)
            tmax = args.twidth*(i+0.5)
            if i == args.tbins-1:
                tmax = 9e9
            emin = args.ewidth*j
            emax = args.ewidth*(j+1)
            if j == args.ebins-1:
                emax = 9e9
            cut = (a["edep"] > emin) & (a["edep"] < emax) & (a["earlytot"] > tmin) & (a["earlytot"] < tmax)
            cuttdrift = a["tdrift"][cut]
            if len(cuttdrift) < args.mincount:
                means[i][j] = default_drift
                stds[i][j] = default_std
            else:
                means[i][j] = np.mean(cuttdrift)
                stds[i][j] = np.std(cuttdrift)
            resids.append(np.ones(len(cuttdrift))*means[i][j]-cuttdrift)
            residerrs.append(np.ones(len(cuttdrift))*stds[i][j])
            ibins.append(np.ones(len(cuttdrift))*i)
            jbins.append(np.ones(len(cuttdrift))*j)

    resids = np.concatenate(resids)
    residerrs = np.concatenate(residerrs)
    ibins = np.concatenate(ibins)
    jbins = np.concatenate(jbins)

    print("Starting calibration:")
    startcut = np.abs(a["utresid"]) < 50
    print("Mean residual:",np.mean(a["utresid"][startcut]))
    print("residual RMS :",np.std(a["utresid"][startcut]))
    print("pull RMS     :",np.std(a["utresid"][startcut]/np.sqrt(a["utresidmvar"][startcut])))
    print("Updated calibration:")
    print("Mean residual:",np.mean(resids))
    print("residual RMS :",np.std(resids))
    print("pull RMS     :",np.std(resids/residerrs))

    # write output
    fout = open(args.output + ".fcl","w")
    fout.write("services.ProditionsService.strawResponse.totTBins : %d\n" % args.tbins)
    fout.write("services.ProditionsService.strawResponse.totTBinWidth : %f\n" % args.twidth)
    fout.write("services.ProditionsService.strawResponse.totEBins : %d\n" % args.ebins)
    fout.write("services.ProditionsService.strawResponse.totEBinWidth : %f\n" % args.ewidth)
    fout.write("services.ProditionsService.strawResponse.totDriftTime : [\n")
    for i in range(args.tbins):
        for j in range(args.ebins):
            fout.write("%.2f" % means[i][j])
            if i < args.tbins-1 or j < args.ebins-1:
                fout.write(", ")
            else:
                fout.write("]")
        fout.write("\n")
    fout.write("services.ProditionsService.strawResponse.totDriftError : [\n")
    for i in range(args.tbins):
        for j in range(args.ebins):
            fout.write("%.2f" % stds[i][j])
            if i < args.tbins-1 or j < args.ebins-1:
                fout.write(", ")
            else:
                fout.write("]")
        fout.write("\n")
    print('Wrote calibrations')

    if args.plot:
        import ROOT

        hval = ROOT.TH1F("hval","hval",100,-50,50)
        ht2d = ROOT.TH2F("ht2d","ht2d",100,-50,50,args.tbins,0,args.tbins*args.twidth)
        he2d = ROOT.TH2F("he2d","he2d",100,-50,50,args.ebins,0,args.ebins*args.ewidth)
        hpull = ROOT.TH1F("hpull","hpull",100,-5,5)
        for i in range(len(resids)):
            hval.Fill(resids[i])
            ht2d.Fill(resids[i],ibins[i]*args.twidth)
            he2d.Fill(resids[i],jbins[i]*args.ewidth)
            hpull.Fill(resids[i]/residerrs[i])
 
        hpull.Draw()
        hpull.GetXaxis().SetTitle("Residual / residual error")
        input()
        htp = ht2d.ProfileY("htp",1,-1,"S")
        htp.GetXaxis().SetTitle("TOT (ns)")
        htp.GetYaxis().SetTitle("Mean drift time residual (ns)")
        htp.Draw()
        input()
        hep = he2d.ProfileY("hep",1,-1,"S")
        hep.GetXaxis().SetTitle("edep (KeV)")
        hep.GetYaxis().SetTitle("Mean drift time residual (ns)")
        hep.Draw()
        input()
        hval.GetXaxis().SetTitle("Drift time residual (ns)")
        hval.Draw()
        input()
