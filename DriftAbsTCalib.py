import awkward as ak
import uproot
import numpy as np
import sys
import argparse

import dbUtility

ECUT = 0.0001
DTCUT = 30
DTRANGE = 12

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run calibration')
    parser.add_argument('--output','-o',help='output filename stub (<output>.fcl, <output>.txt)',default='output')
    parser.add_argument('--dbpurpose','-p',help='database purpose',default='')
    parser.add_argument('--dbversion','-v',help='database version',default='')
    parser.add_argument('--dbtext','-t',help='comma separated list of database text files',default='')
    parser.add_argument('--source','-s',help='input SHD filename',default='')
    parser.add_argument('--source-list','-S',help='list of input SHD filenames',default='')

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
    for batch in uproot.iterate([fn + ":EventNtuple/ntuple" for fn in fns],filter_name=["trkhits","evtinfo/*"]):
        cut = (batch.trkhits.edep > ECUT) & (batch.trkhits.state > -2) & (np.abs(batch.trkhits.udoca) < 2.5) & (np.abs(batch.trkhits.udoca) > 0.5)
        planes.append(np.copy(ak.flatten(ak.flatten(batch.trkhits.plane[cut])).to_numpy()))
        panels.append(np.copy(ak.flatten(ak.flatten(batch.trkhits.panel[cut])).to_numpy()))
        straws.append(np.copy(ak.flatten(ak.flatten(batch.trkhits.straw[cut])).to_numpy()))

        udts.append(np.copy(ak.flatten(ak.flatten(batch.trkhits.rdrift[cut]-np.abs(batch.trkhits.udoca[cut]))).to_numpy()))
        run = int(batch["run"][0])
        print(len(udts[-1])*len(udts))

    print("Read data")
    
    c = dbUtility.getCalibrations(run,["TrkPreampStraw"],args.dbpurpose,args.dbversion,args.dbtext)
    print("Read calibrations")

    planes = np.concatenate(planes)
    panels = np.concatenate(panels)
    straws = np.concatenate(straws)
    udts = np.concatenate(udts)
    sids = planes*6*96+panels*96+straws
    counts = np.bincount(sids,minlength=36*6*96)
    expected_counts = np.median(counts[counts != 0])
    goodchans = (counts > expected_counts/5) & (counts < expected_counts*3)
    
    #FIXME switch to sorting first to avoid inefficiency
    offsets = np.zeros(36*6*96)
    for ipl in range(36):
        cut1 = (planes == ipl)
        udts1 = udts[cut1]
        panels1 = panels[cut1]
        straws1 = straws[cut1]
        for ipa in range(6):
            cut2 = (panels1 == ipa)
            udts2 = udts1[cut2]
            straws2 = straws1[cut2]
            for istraw in range(96):
                cut3 = (straws2 == istraw)
                udts3 = udts2[cut3]
                i = ipl*6*96 + ipa*96 + istraw
                if goodchans[i] and len(udts3) > 50:
                    offsets[i] = np.mean(udts3)/0.0625
    print("Mean offset of %.2f" % (np.mean(offsets[offsets != 0])))
    offsets[offsets != 0] -= np.mean(offsets[offsets != 0])
    

    for i in range(36*6*96):
        c["TrkPreampStraw"]["delay_cal"][i] = "%f" % (float(c["TrkPreampStraw"]["delay_cal"][i])-offsets[i])
        c["TrkPreampStraw"]["delay_hv"][i] = "%f" % (float(c["TrkPreampStraw"]["delay_hv"][i])-offsets[i])
    dbUtility.writeCalibrations(c,["TrkPreampStraw"],args.output)
    print('Wrote %d calibrations, avg mag = %f' % (len(offsets[offsets != 0]),np.mean(np.abs(offsets[offsets != 0]))))
