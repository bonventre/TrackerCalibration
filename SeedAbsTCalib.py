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
    parser.add_argument('--ecut','-e',type=float,help='energy cut (MeV)',default=0.0001)
    parser.add_argument('--mint','-m',type=float,help='minimum udt (ns)',default=-20)
    parser.add_argument('--maxt','-M',type=float,help='maximum udt (ns)',default=70)
    parser.add_argument('--mincount','-c',type=int,help='minimum hit count',default=30)
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
    
    planes = []
    panels = []
    straws = []
    udts = []
    run = 0
    for batch in uproot.iterate([fn + ":EventNtuple/ntuple" for fn in fns],filter_name=["trkhits","evtinfo/*","trk/*"]):
        cut = (batch.trkhits.edep > args.ecut) & (batch.trkhits.state > -2) & (batch.trkhits.udt > args.mint) & (batch.trkhits.udt < args.maxt)
        planes.append(np.copy(ak.flatten(ak.flatten(batch.trkhits.plane[cut])).to_numpy()))
        panels.append(np.copy(ak.flatten(ak.flatten(batch.trkhits.panel[cut])).to_numpy()))
        straws.append(np.copy(ak.flatten(ak.flatten(batch.trkhits.straw[cut])).to_numpy()))
        udts.append(np.copy(ak.flatten(ak.flatten(batch.trkhits.udt[cut])).to_numpy()))
        run = int(batch["run"][0])

    print("Read data")
    
    c = dbUtility.getCalibrations(run,["TrkPreampStraw","TrkDelayPanel"],args.dbpurpose,args.dbversion,args.dbtext)
    print("Read calibrations")

    planes = np.concatenate(planes)
    panels = np.concatenate(panels)
    straws = np.concatenate(straws)
    udts = np.concatenate(udts)

    sids = planes*6*96+panels*96+straws
    order = ak.argsort(sids)
    sids = sids[order]
    runs = ak.run_lengths(sids)
#    planes = ak.unflatten(planes[order],runs)
#    panels = ak.unflatten(panels[order],runs)
#    straws = ak.unflatten(straws[order],runs)
    udts = ak.unflatten(udts[order],runs)
    sids = ak.unflatten(sids,runs)

    counts = ak.count(sids,axis=1)
    expected_counts = np.median(counts[counts != 0])
    goodstraw = (counts > expected_counts/5) & (counts < expected_counts*3)
    means = np.mean(udts,axis=1) 
    print("%d counts per straw" % (np.mean(counts[goodstraw])))

    offsets = np.zeros(36*6*96)
    for i in range(len(goodstraw)):
        if goodstraw[i] and len(udts[i]) > args.mincount:
            offsets[sids[i][0]] = means[i]

    print('Subtracting overall average offset of %f' % np.mean(offsets[offsets != 0]))
    offsets[offsets != 0] -= np.mean(offsets[offsets != 0])
    # subtract panel level averages
    paneloffsets = np.zeros(36*6)
    for i in range(len(paneloffsets)):
        temp = offsets[i*96:(i+1)*96]
        if len(temp[temp != 0]) > 0:
            paneloffsets[i] = np.mean(temp[temp != 0])
            temp[temp != 0] -= paneloffsets[i]
    print('Mean panel offsets: %f' % (np.mean(paneloffsets[paneloffsets != 0])))


    for i in range(36*6*96):
        c["TrkPreampStraw"]["delay_cal"][i] = "%f" % (float(c["TrkPreampStraw"]["delay_cal"][i])-offsets[i])
        c["TrkPreampStraw"]["delay_hv"][i] = "%f" % (float(c["TrkPreampStraw"]["delay_hv"][i])-offsets[i])
    for i in range(36*6):
        c["TrkDelayPanel"]["delay"][i] = "%f" % (float(c["TrkDelayPanel"]["delay"][i])-paneloffsets[i])
    dbUtility.writeCalibrations(c,["TrkPreampStraw","TrkDelayPanel"],args.output + ".txt")
    print('Wrote %d calibrations, avg mag = %f' % (np.sum(goodstraw),np.mean(np.abs(offsets[offsets != 0]))))

    if args.plot:
        full_sids = ak.flatten(sids)
        full_udts = ak.flatten(udts)
        full_offsets = offsets[full_sids] + paneloffsets[full_sids//96]
        corrected_udts = full_udts - full_offsets
    
        import ROOT
        h0 = ROOT.TH1F("h0","h0",400,-50,150)
        h1 = ROOT.TH1F("h1","h1",400,-50,150)
        h0.FillN(len(full_udts),full_udts.to_numpy().astype(np.double),np.ones(len(full_udts),dtype=np.double))
        h1.FillN(len(full_udts),corrected_udts.to_numpy().astype(np.double),np.ones(len(full_udts),dtype=np.double))

        h0.GetXaxis().SetTitle("Delta t (ns)")
        h1.SetLineColor(ROOT.kRed)

        l = ROOT.TLegend(0.65,0.65,0.85,0.85)
        l.AddEntry(h0,"Initial","l")
        l.AddEntry(h1,"Calibrated","l")
    
        h0.Draw()
        h1.Draw("same")
        l.Draw()
        input()
