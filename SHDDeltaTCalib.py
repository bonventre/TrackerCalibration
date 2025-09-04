import awkward as ak
import uproot
import numpy as np
import sys
import argparse

import dbUtility

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run calibration')
    parser.add_argument('--output','-o',help='output filename stub (<output>.fcl, <output>.txt)',default='output')
    parser.add_argument('--dbpurpose','-p',help='database purpose',default='')
    parser.add_argument('--dbversion','-v',help='database version',default='')
    parser.add_argument('--dbtext','-t',help='comma separated list of database text files',default='')
    parser.add_argument('--source','-s',help='input SHD filename',default='')
    parser.add_argument('--source-list','-S',help='list of input SHD filenames',default='')
    parser.add_argument('--dtcut','-d',type=float,help='initial dt cut (ns)',default=30.0)
    parser.add_argument('--dtrange','-r',type=float,help='dt range (ns)',default=12.0)
    parser.add_argument('--ecut','-e',type=float,help='energy cut (MeV)',default=0.0001)
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

    a = {field: [] for field in ["tcal","thv","sid"]}
    counts = 0
    runid = 0
    for batch in uproot.iterate([fn + ":StrawHitDiagnostics/shdiag" for fn in fns],filter_name=["time","edep","plane","panel","straw","runid"]):
        batch = batch[batch.edep > args.ecut]
        a["tcal"].append(np.copy(batch.time.tcal.to_numpy()))
        a["thv"].append(np.copy(batch.time.thv.to_numpy()))
        a["sid"].append(np.copy((batch.plane*96*6+batch.panel*96+batch.straw).to_numpy()))
        runid = batch.runid[0]
        counts += len(batch.edep)
        print(counts)

    for field in a:
        a[field] = np.concatenate(a[field])
    order = ak.argsort(a["sid"])
    for field in a:
        a[field] = a[field][order]
    run_lengths = ak.run_lengths(a["sid"])
    dts = ak.unflatten(a["tcal"]-a["thv"],run_lengths)
    sids = ak.unflatten(a["sid"],run_lengths)
    
    print("Read data")
    
    c = dbUtility.getCalibrations(int(runid),["TrkPreampStraw"],args.dbpurpose,args.dbversion,args.dbtext)
    print("Read calibrations")

    dts = dts[np.abs(dts) < args.dtcut]
    counts = ak.count(dts,axis=1)
    means0 = np.mean(dts,axis=1)
    dts = dts[(dts > means0-args.dtrange) & (dts < means0+args.dtrange)]
    means1 = np.mean(dts,axis=1)

    expected_counts = np.median(counts[counts != 0])
    goodstraw = (counts > expected_counts/5) & (counts < expected_counts*3)

    offsets = np.zeros(36*6*96)
    for i in range(len(goodstraw)):
        if goodstraw[i]:
            offsets[sids[i][0]] = means1[i]

    for i in range(36*6*96):
        c["TrkPreampStraw"]["delay_cal"][i] = "%f" % (float(c["TrkPreampStraw"]["delay_cal"][i])-offsets[i]/2.)
        c["TrkPreampStraw"]["delay_hv"][i] = "%f" % (float(c["TrkPreampStraw"]["delay_hv"][i])+offsets[i]/2.)
    dbUtility.writeCalibrations(c,["TrkPreampStraw"],args.output + ".txt")
    print('Wrote %d calibrations, avg mag = %f' % (np.sum(goodstraw),np.mean(np.abs(offsets[offsets != 0]))))

    if args.plot:
        full_offsets = offsets[a["sid"]]
        full_dts = a["tcal"]-a["thv"]
        corrected_dts = full_dts - full_offsets
    
        import ROOT
        h0 = ROOT.TH1F("h0","h0",300,-30,30)
        h1 = ROOT.TH1F("h1","h1",300,-30,30)
        h0.FillN(len(full_dts),full_dts.astype(np.double),np.ones(len(full_dts),dtype=np.double))
        h1.FillN(len(corrected_dts),corrected_dts.astype(np.double),np.ones(len(corrected_dts),dtype=np.double))

        h0.GetXaxis().SetTitle("Delta t (ns)")
        h1.SetLineColor(ROOT.kRed)

        l = ROOT.TLegend(0.65,0.65,0.85,0.85)
        l.AddEntry(h0,"Initial","l")
        l.AddEntry(h1,"Calibrated","l")
    
        h0.Draw()
        h1.Draw("same")
        l.Draw()
        input()
