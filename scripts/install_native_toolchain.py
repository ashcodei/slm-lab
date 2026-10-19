"""Install a hash-verified official Zig release and warm a trusted C++ cache."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request


def install(root,version='0.17.0'):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    metadata=root/'native-toolchain.json'
    if metadata.exists():
        record=json.loads(metadata.read_text())
        if record['version']!=version:raise ValueError('Installed compiler version differs; use a separate tool directory')
        if hashlib.sha256(Path(record['compiler_path']).read_bytes()).hexdigest()!=record['compiler_sha256']:raise ValueError('Installed compiler changed')
        return record
    index=json.load(urllib.request.urlopen('https://ziglang.org/download/index.json',timeout=30))
    package=index[version]['x86_64-linux'];archive=root/Path(package['tarball']).name
    with urllib.request.urlopen(package['tarball'],timeout=30) as response,archive.open('wb') as target:
        while block:=response.read(1<<20):target.write(block)
    if archive.stat().st_size!=int(package['size']) or hashlib.sha256(archive.read_bytes()).hexdigest()!=package['shasum']:
        raise ValueError('Official compiler archive size/hash mismatch')
    with tarfile.open(archive) as bundle:
        names={member.name.split('/')[0] for member in bundle.getmembers()}
        if len(names)!=1:raise ValueError('Unexpected toolchain archive layout')
        bundle.extractall(root,filter='data')
    directory=root/next(iter(names));compiler=directory/'zig'
    work=root/'native-warm';work.mkdir(exist_ok=True)
    source=work/'hello.cpp';source.write_text('#include <iostream>\nint main(){std::cout << 42 << std::endl;}\n')
    cache=root/'native-compiler-cache';cache.mkdir(exist_ok=True)
    environment={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','ZIG_GLOBAL_CACHE_DIR':str(cache),'ZIG_LOCAL_CACHE_DIR':str(work/'local')}
    subprocess.run([str(compiler),'c++','-target','x86_64-linux-musl','-static','-std=c++17','-O2','-fno-lto',
                    str(source),'-o',str(work/'hello')],check=True,env=environment)
    completed=subprocess.run([str(work/'hello')],capture_output=True,text=True,check=True)
    if completed.stdout.strip()!='42':raise ValueError('Trusted compiler warmup output mismatch')
    record={'compiler':'Zig C++ frontend','version':version,'url':package['tarball'],'archive_sha256':package['shasum'],
            'compiler_path':str(compiler),'toolchain_root':str(directory),
            'compiler_sha256':hashlib.sha256(compiler.read_bytes()).hexdigest(),'cache_path':str(cache)}
    metadata.write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='.tools')
    parser.add_argument('--version',default='0.17.0');args=parser.parse_args()
    print(json.dumps(install(args.root,args.version),indent=2))
