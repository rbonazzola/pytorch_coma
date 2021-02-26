import pandas as pd

def get_dosages(snpfile):
  rs = pd.read_csv(snpfile, skiprows=4, sep="\t")
  rs  = rs.set_index(rs.ID).iloc[:,9:]
  rs = rs.applymap(lambda x: x.split(":")[1].split(",").index('1'))
  samples = rs.columns
  rs = rs.values
  return {samples[i]: rs[0,i] for i,_ in enumerate(samples)}