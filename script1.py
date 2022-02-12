#SCRIPT1 DATASET CREATION 
import pandas as pd
import math



#FUNCTION USED FOR SAMPLING THE DATAFRAME TO CREATE A BALANCED DATASET OF SUBCLASSES // SPECIAL MINORITY SUBCLASSES AND MAJORITY SUBCLASSES
def sample(list_of_names,df,num):
  result = pd.DataFrame()
  for item in list_of_names:
#IF STATEMENT TO RETRIEVE ALL SAMPLES OF MINORITY SUBCLASSSES
    if item == 'worm' or item == 'udpstorm' or item == 'httptunnel' or item == 'land' or item == 'spy' or item == 'xsnoop' or item == 'phf' or item == 'xlock' or item == 'ftp_write' or item == 'imap' or item == 'sendmail' or item == 'named' or item == 'multihop':
     df_temp = df.loc[(df['class'] == item)]
     result = result.append(df_temp, ignore_index = True)
#ELSE SAMPLE SPECIFIED NUMBER OF SAMPLES FOR A TOTAL SUM OF num PER CLASS
    else:
     df_temp = df.loc[(df['class'] == item)]
     temp = df_temp.sample(num)
     result = result.append(temp, ignore_index = True)
  return result 

#INITIAL CSVs THAT I WILL CONCATENATE
url = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/KDDTrain.txt'
url2 = 'https://raw.githubusercontent.com/jasonalexander87/machineL/main/KDDTest.txt'

#url = 'KDDTrain.txt'
#url2 = 'KDDTest.txt'

#FEATURES NAMES FOR CSV COLUMNS
headerList = ['duration', 'protocol_type', 'service', 'flag','src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
              'num_failed_logins', 'logged_in','num_compromised', 'root_shell', 'su_attempted', 'num_root','num_file_creations',
              'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
              'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
              'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
              'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class','difficulty']

#CREATE DATAFRAMES OF CSVs
df = pd.read_csv(url,names=headerList)
df2 = pd.read_csv(url2,names=headerList)

#df['class'].value_counts()
#df3 = df2.loc[(df2['class'] == 'buffer_overflow') | (df2['class'] == 'loadmodule') | (df2['class'] == 'rootkit') | (df2['class'] == 'perl') | (df2['class'] == 'sqlattack') | (df2['class'] == 'xterm') | (df2['class'] == 'ps')]

#CONCAT THE TWO DATAFRAMES
df = df.append(df2, ignore_index = True)
#print(df['class'].value_counts())
#WS EDW OLA KALA
#print(df['class'].unique())

#DROP DIFFICULTY COLUMN HAS TO BE DONE IRRELEVANT INFORMATION ABOUT ENTRY
df.drop('difficulty', inplace=True, axis=1)

#ANNOTATE SUBCLASSES
#U2R
df.loc[(df['class'] == 'buffer_overflow') | (df['class'] == 'loadmodule') | (df['class'] == 'rootkit') | (df['class'] == 'perl') | (df['class'] == 'sqlattack') | (df['class'] == 'xterm') | (df['class'] == 'ps'), 'CATEGORY'] = 'U2R'

#NORMAL
df.loc[(df['class'] == 'normal'), 'CATEGORY'] = 'NORMAL'

#DDOS
df.loc[ (df['class'] == 'mailbomb') |(df['class'] == 'back') | (df['class'] == 'land') | (df['class'] == 'neptune') | (df['class'] == 'pod') | (df['class'] == 'smurf') | (df['class'] == 'teardrop') | (df['class'] == 'apache2') | (df['class'] == 'udpstorm') | (df['class'] == 'processtable') | (df['class'] == 'worm'), 'CATEGORY'] = 'DDOS'

#PROBE
df.loc[(df['class'] == 'satan') | (df['class'] == 'ipsweep') | (df['class'] == 'nmap') | (df['class'] == 'portsweep') | (df['class'] == 'mscan') | (df['class'] == 'saint'), 'CATEGORY'] = 'PROBE'

#R2L
df.loc[(df['class'] == 'guess_passwd') | (df['class'] == 'ftp_write') | (df['class'] == 'imap') | (df['class'] == 'phf') | (df['class'] == 'multihop') | (df['class'] == 'warezmaster') | (df['class'] == 'warezclient') | (df['class'] == 'spy') | (df['class'] == 'xlock') | (df['class'] == 'xsnoop') | (df['class'] == 'snmpguess') | (df['class'] == 'snmpgetattack') | (df['class'] == 'httptunnel') | (df['class'] == 'sendmail') | (df['class'] == 'named'), 'CATEGORY'] = 'R2L'

#df_temp = df.loc[(df['CATEGORY'] == 'R2L')]
#print(df_temp['class'].value_counts())


#SAMPLING
#FOR PROBE CATEGORY
temp = df.loc[(df['CATEGORY'] == 'PROBE')]
names_probe = temp['class'].unique().tolist()
df_total_probe = sample(names_probe,df,170)

#FOR DDOS CATEGORY
temp = df.loc[(df['CATEGORY'] == 'DDOS')]
names_ddos = temp['class'].unique().tolist()
df_total_ddos = sample(names_ddos,df,130)

#FOR R2L CATEGORY
temp = df.loc[(df['CATEGORY'] == 'R2L')]
names_r2l = temp['class'].unique().tolist()
df_total_r2l = sample(names_r2l,df,170)

#FOR NORMAL CLASS 
df_total_normal = df.loc[(df['CATEGORY'] == 'NORMAL')].sample(1000)

#FOR U2R MINORITY CLASS ALL SAMPLES INCLUDED AND CONCATENATE WITH REST
df_total_u2r = df.loc[(df['CATEGORY'] == 'U2R')]

df_total = pd.DataFrame()
df_total = df_total.append(df_total_probe, ignore_index = True)
df_total = df_total.append(df_total_ddos, ignore_index = True)
df_total = df_total.append(df_total_r2l, ignore_index = True)
df_total = df_total.append(df_total_normal, ignore_index = True)
df_total = df_total.append(df_total_u2r, ignore_index = True)


df_total.to_csv('dataset_Final.csv', index=False)
