"""Download and hash local semantic assets from public official/ModelScope sources."""
import argparse, concurrent.futures, hashlib, json, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode

# Verified released bytes; reject incomplete existing official files.
OFFICIAL_ASSETS = {'dinov2_vitl14_reg4_pretrain.pth': (1217607321, '36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51'), 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth': (101317814, '706fba5bcb7019d80e127f0ff123872a75e0e6099527cf1911db85bd61c88ef5'), 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth': (2152639923, '13378274c79f67af8b47b11dd57827d659e7b1ee01e4dde58d2266422ea7846a'), 'bpe_simple_vocab_16e6.txt.gz': (1356917, '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a')}

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

def fetch(url, path, size=None, sha=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        digest=sha256_file(path)
        if (size is None or path.stat().st_size==size) and (sha is None or sha==digest):
            return dict(path=str(path),sha256=digest,bytes=path.stat().st_size,source=url)
    part=path.with_suffix(path.suffix+'.part')
    for attempt in range(5):
        offset=part.stat().st_size if part.exists() else 0
        try:
            with urlopen(Request(url,headers={'Range':f'bytes={offset}-'} if offset else {}),timeout=60) as r:
                resume=bool(offset and r.status==206)
                if not resume: offset=0
                total=size or (int(r.headers['Content-Length'])+offset if r.headers.get('Content-Length') else None)
                milestone=offset//(256*1024**2)
                with part.open('ab' if resume else 'wb') as f:
                    for block in iter(lambda:r.read(4*1024**2),b''):
                        f.write(block);offset+=len(block)
                        if offset//(256*1024**2)>milestone:
                            milestone=offset//(256*1024**2)
                            print(path.name,round(offset/1024**2),'MiB',flush=True)
            if total and total!=offset: raise ValueError('incomplete download')
            digest=sha256_file(part)
            if sha and sha!=digest:
                part.unlink();raise ValueError('SHA256 mismatch')
            part.replace(path)
            print('verified',path.name,digest,flush=True)
            return dict(path=str(path),sha256=digest,bytes=offset,source=url)
        except Exception as e:
            print('retry',path.name,type(e).__name__,str(e)[:90],flush=True)
            if attempt==4:raise
            time.sleep(2**attempt)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root.resolve();jobs=[]
    for name in ['dinov2_vitl14_reg4_pretrain.pth',
                 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth',
                 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth']:
        jobs.append((f'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/{name}',root/name,*OFFICIAL_ASSETS[name]))
    jobs.append(('https://dl.fbaipublicfiles.com/dinov2/thirdparty/bpe_simple_vocab_16e6.txt.gz',
                 root/'bpe_simple_vocab_16e6.txt.gz',*OFFICIAL_ASSETS['bpe_simple_vocab_16e6.txt.gz']))
    for model,names,folder in [
        ('ciqiangxu/DINOv3',{'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
                            'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth'},''),
        ('google/siglip2-base-patch16-224',None,'siglip2-base-patch16-224')]:
        with urlopen(f'https://modelscope.cn/api/v1/models/{model}/repo/files?Revision=master&Recursive=true',timeout=30) as r:
            files=json.load(r)['Data']['Files']
        for f in files:
            name=f['Path']
            if names is not None and name not in names:continue
            if names is None and Path(name).suffix not in ('.json','.model','.safetensors','.md'):continue
            query=urlencode({'Revision':f['Revision'],'FilePath':name})
            jobs.append((f'https://modelscope.cn/api/v1/models/{model}/repo?{query}',
                         root/folder/name,f['Size'],f.get('Sha256')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda args:fetch(*args),jobs))
    (root/'assets-manifest.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
