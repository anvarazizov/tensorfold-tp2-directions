#!/usr/bin/env python3
"""Run explicit HTTP requests only after the destination pair is safely started."""
import argparse,hashlib,json,os,pathlib,struct,time,urllib.error,urllib.request
SET='pqhaz-1017-projected-experimental';SHA='097fff6bf88de26730240d1bc20747fb63cc488db050807a6e8efce7452d763c';MASK=sum(1<<i for i in range(2,40))
def packet(ablation):
 if ablation is None or 'direction_set' not in ablation:return bytes(44).hex()
 mask=sum(1<<i for i in ablation.get('layers',range(2,40)))
 return (struct.pack('<fQ',ablation.get('strength',0),mask)+bytes.fromhex(SHA)).hex()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--warm-cache',action='store_true');p.add_argument('--api-key-env',default='TF_API_KEY');a=p.parse_args();out=pathlib.Path(a.output);assert not out.exists(),'Use a fresh output file'
 rows=[];headers={'Content-Type':'application/json'}
 if os.environ.get(a.api_key_env):headers['Authorization']='Bearer '+os.environ[a.api_key_env]
 def request(body,path='/chat/completions'):
  start=time.time();req=urllib.request.Request(a.base_url.rstrip('/')+path,data=json.dumps(body).encode(),headers=headers)
  try:
   with urllib.request.urlopen(req,timeout=300) as r:return r.status,json.load(r),time.time()-start
  except urllib.error.HTTPError as e:return e.code,e.read().decode(),time.time()-start
 def save():out.write_text(json.dumps({'complete':False,'rows':rows},indent=2))
 def run(name,ablation=None,prompt='Return only the integer: 17 + 25.',salt='portable-direction-acceptance-v1'):
  b={'model':a.model,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':42,'max_tokens':64,'reasoning_effort':'none','cache_salt':salt}
  if ablation is not None:b['ablation']=ablation
  status,value,seconds=request(b);valid=status==200 and isinstance(value,dict) and value.get('tensorfold',{}).get('directions_packet_hex')==packet(ablation)
  rows.append({'name':name,'status':status,'seconds':seconds,'expected_packet_hex':packet(ablation),'passed':valid,'response':value});save();assert valid,(name,status,value)
  return value
 base=run('OFF-absent');assert base['choices'][0]['message']['content'].strip()=='42','Baseline arithmetic failed'
 disabled=run('OFF-explicit',{'enabled':False,'strength':0});assert base['tensorfold']['token_sha256']==disabled['tensorfold']['token_sha256'],'OFF mismatch'
 on=run('ON-half',{'direction_set':SET,'enabled':True,'strength':.5});again=run('OFF-return');assert base['tensorfold']['token_sha256']==again['tensorfold']['token_sha256'],'A/B/A did not return to OFF'
 run('ON-selected-layers',{'direction_set':SET,'enabled':True,'strength':.5,'layers':[2,3]})
 for name,bad in [('strength-out-of-range',{'enabled':True,'direction_set':SET,'strength':2}),('duplicate-layers',{'enabled':True,'direction_set':SET,'strength':.5,'layers':[2,2]}),('unknown-set',{'enabled':True,'direction_set':'missing','strength':.5}),('nonboolean-enabled',{'enabled':1})]:
  status,value,seconds=request({'model':a.model,'messages':[{'role':'user','content':'Hi'}],'max_tokens':8,'ablation':bad});rows.append({'name':name,'passed':status==400,'status':status,'response':value});save();assert status==400,(name,status)
 for path,extra in [('/responses',{'input':'Say hi'}),('/messages',{'messages':[{'role':'user','content':'Say hi'}],'max_tokens':8})]:
  status,value,seconds=request({'model':a.model,'ablation':{},**extra},path);passed=status==400 and 'ablation' in str(value);rows.append({'name':'reject'+path,'passed':passed,'status':status,'response':value});save();assert passed,(path,status,value)
 for alpha in [0,.5]:
  ablation={'direction_set':SET,'enabled':alpha!=0,'strength':alpha};body={'model':a.model,'messages':[{'role':'user','content':'Return only the integer: 17 + 25.'}],'temperature':0,'seed':42,'max_tokens':64,'reasoning_effort':'none','stream':True,'ablation':ablation}
  req=urllib.request.Request(a.base_url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers=headers)
  with urllib.request.urlopen(req,timeout=180) as response:raw=response.read().decode()
  events=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line[6:]!='[DONE]'];metadata=[e['tensorfold'] for e in events if 'tensorfold' in e];text=''.join(c.get('delta',{}).get('content','') or '' for e in events for c in e.get('choices',[]))
  passed='[DONE]' in raw and bool(metadata) and metadata[-1].get('directions_packet_hex')==packet(ablation) and text.strip()=='42';rows.append({'name':'SSE-'+str(alpha),'passed':passed,'text':text,'events':events});save();assert passed,'SSE packet/content mismatch'
 if a.warm_cache:
  prompt=('This is an ordinary archived sentence about trees rivers books and clouds.\n'*700)+'\nThe code is HARBOR73. Return only the code.';refs={}
  salt = 'direction-cache-test-' + str(time.time_ns())
  for i,alpha in enumerate([0,0,.5,.5,0,0]):
   v=run('warm-'+str(i),{'direction_set':SET,'enabled':alpha!=0,'strength':alpha},prompt,salt)
   actual=v['tensorfold']['token_sha256'];assert alpha not in refs or refs[alpha]==actual,'Warm-cache token identity mismatch';refs.setdefault(alpha,actual)
   assert v['choices'][0]['message']['content'].strip()=='HARBOR73','Warm retrieval failed'
   if i in [0,2]:assert v['usage']['prompt_tokens_details']['cached_tokens']==0,'Unexpected cache reuse for a new mode'
   if i in [1,3,5]:assert v['usage']['prompt_tokens_details']['cached_tokens']>0,'No actual warm snapshot observed'
 out.write_text(json.dumps({'complete':True,'passed':True,'tests':len(rows),'warm_cache_exercised':a.warm_cache,'rows':rows,'limits':'Protocol/cache isolation check, not proof of broad capability or refusal suppression; target GPU numerical test and task evaluation still required.'},indent=2));print(json.dumps({'passed':True,'tests':len(rows),'output':str(out)}))
if __name__=='__main__':main()
