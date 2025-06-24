import awkward as ak
import uproot
import ROOT
import sys
import numpy as np

# usage: python TOTDriftTime.py <name of root file with EventNtuple ttree> <num energy bins> <energy binning (KeV)> <num tot bins> <TOT binning (ns)> [1=usemc/0=usedata]

fn = sys.argv[1]
pcode = int(sys.argv[2])
ebins = int(sys.argv[3])
ewidth = float(sys.argv[4])
tbins = int(sys.argv[5])
twidth = float(sys.argv[6])
usemc = 0
if len(sys.argv) > 7:
  usemc = int(sys.argv[7])

MINCOUNT = 10

fields = ["tot","etime","edep","state","udt"]
mcfields = ["startCode","earlyend","t0","tprop"]
a = {field : [] for field in fields}
if usemc:
    for field in mcfields:
        a[field] = []
for batch in uproot.iterate([fn + ":EventNtuple/ntuple"],filter_name=["trkhitsmc","trkhits","trk.fitcon","trk.nactive","trk.nhits"]):
    for field in fields:
        a[field].append(np.copy(ak.flatten(ak.flatten(batch["trkhits"][field])).to_numpy()))
    if usemc:
        for field in mcfields:
            a[field].append(np.copy(ak.flatten(ak.flatten(batch["trkhitsmc"][field][ak.local_index(batch["trkhitsmc"][field]) < ak.num(batch["trkhits"][fields[0]],axis=2)])).to_numpy()))

for field in a:
    a[field] = np.concatenate(a[field])

if usemc:
    cut = (a["startCode"] == pcode) & (a["state"] > -2)
else:
    cut = (a["state"] > -2) & (a["udt"] > -10) & (a["udt"] < 60)
for field in a:
    a[field] = a[field][cut]
if not usemc:
    a["earlyend"] = np.argmin(a["etime"],axis=1)

a["earlytime"] = np.where(a["earlyend"] == 0,a["etime"][:,0],a["etime"][:,1])
a["earlytot"] = np.where(a["earlyend"] == 0,a["tot"][:,0],a["tot"][:,1])
if usemc:
    a["tdrift"] = a["earlytime"] - a["t0"] - a["tprop"]
else:
    a["tdrift"] = a["udt"]

default_drift = np.mean(a["tdrift"])
default_std = np.std(a["tdrift"])
means = np.zeros((tbins,ebins))
stds = np.zeros((tbins,ebins))

resids = []
residerrs = []

hval = ROOT.TH1F("hval","hval",100,-50,50)
ht2d = ROOT.TH2F("ht2d","ht2d",100,-50,50,tbins,0,tbins*twidth)
he2d = ROOT.TH2F("he2d","he2d",100,-50,50,ebins,0,ebins*ewidth)
hpull = ROOT.TH1F("hpull","hpull",100,-5,5)

for i in range(tbins):
    for j in range(ebins):
        tmin = twidth*(i-0.5)
        tmax = twidth*(i+0.5)
        if i == tbins-1:
            tmax = 9e9
        emin = ewidth*j
        emax = ewidth*(j+1)
        if j == ebins-1:
            emax = 9e9
        cut = (a["edep"] > emin) & (a["edep"] < emax) & (a["earlytot"] > tmin) & (a["earlytot"] < tmax)
        cuttdrift = a["tdrift"][cut]
        if len(cuttdrift) < MINCOUNT:
            means[i][j] = default_drift
            stds[i][j] = default_std
        else:
            means[i][j] = np.mean(cuttdrift)
            stds[i][j] = np.std(cuttdrift)
        resids = np.ones(len(cuttdrift))*means[i][j]-cuttdrift
        residerrs = np.ones(len(cuttdrift))*stds[i][j]
        for k in range(len(resids)):
            hval.Fill(resids[k])
            ht2d.Fill(resids[k],i*twidth)
            he2d.Fill(resids[k],j*ewidth)
            hpull.Fill(resids[k]/residerrs[k])
        print("%.2f, " % means[i][j],end='')
    print("")
    
print("totDriftError : [")
for i in range(tbins):
    for j in range(ebins):
        print("%.2f, " % stds[i][j],end='')
    print("")

print("PROCESSING DONE")
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
