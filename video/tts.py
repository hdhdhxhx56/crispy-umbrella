import subprocess, sys
sys.path.insert(0,'.')
from script import SCENES, CHARS
n=0
for si,sc in enumerate(SCENES):
    for li,(who,say,sub,sl) in enumerate(sc['lines']):
        raw=f'a/r{si}_{li}.wav'; out=f'a/{si}_{li}.wav'
        subprocess.run(['RHVoice-test','-p',CHARS[who]['voice'],'-o',raw],input=say.encode(),check=True)
        subprocess.run(['ffmpeg','-y','-loglevel','error','-i',raw,'-ac','1','-ar','24000',out],check=True)
        n+=1
print(n)
