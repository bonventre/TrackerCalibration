from io import StringIO
import csv

import DbService
import DbTables

# FIXME use query info
field_names = {"TrkPreampStraw": "index,delay_hv,delay_cal,threshold_hv,threshold_cal,gain".split(","),
            "TrkDelayPanel": "index,delay".split(","),
            "TrkAlignPanel": "index,strawid,dx,dy,dz,rx,ry,rz".split(",")}

def getCalibrations(run,table_names,purpose="",version="",text_files=[]):
    tables = {}
    if purpose and version:
        dbt = DbService.DbTool()
        dbt.init()
        for tn in table_names:
            try:
                dbt.setArgs(["print-run","--purpose",purpose,"--version",version,"--table",tn,"--run","%d" % run,"--content","1"])
                dbt.run()
                table = dbt.getResult()
                tables[tn] = table
            except Exception as e:
                print(e)

    if text_files:
        dbu = DbTables.DbUtil()
        for fn in text_files:
            t = dbu.readFile(fn)
            for i in range(len(t)):
                if t[i].table().name() in table_names and t[i].iov().inInterval(run,1):
                    table = t[i].table().csv()
                    tables[t[i].table().name()] = table
    tables = {tn: list(csv.reader(StringIO(tables[tn]))) for tn in tables}
    results = {}
    for tn in tables:
        results[tn] = {field: [row[i] for row in tables[tn]] for i, field in enumerate(field_names[tn])}
    return results

def writeCalibrations(calib,table_names,filename):
    fout = open(filename,"w")
    for tn in table_names:
        fout.write("TABLE %s\n" % tn)
        for i in range(len(calib[tn][field_names[tn][0]])):
            line = ""
            for j in range(len(field_names[tn])):
                line += calib[tn][field_names[tn][j]][i]
                if j != len(field_names[tn])-1:
                    line += ","
            line += "\n"
            fout.write(line)
    fout.close()


