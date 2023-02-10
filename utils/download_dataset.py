import subprocess
import os
import zipfile
from argparse import ArgumentParser
import platform

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)

def unzip_dataset(src_fpath, dest_dpath):

  if not os.path.exists(dest_dpath):
    os.makedirs(dest_dpath)

  with zipfile.ZipFile(src_fpath,'r') as zip_ref:
    zip_ref.extractall(dest_dpath)

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument('--savedir', default="./dataset/camvid", help="directory to save the dataset")
    args = parser.parse_args()
    
    print("downloading the dataset ... \n")
    runcmd('wget --content-disposition -p http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip',verbose=True)
    
   
    src_fpath = "./mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/CamSeq01.zip"
    dest_dpath = args.savedir
    
    print(f"Unzipping to {dest_dpath} ... \n") 
    unzip_dataset(src_fpath, dest_dpath)
    
    print("Cleaning ...")
    if platform.system == "Windows":
      runcmd('rd -r ./mi.eng.cam.ac.uk',verbose=True)
    else:
      runcmd('rm -r ./mi.eng.cam.ac.uk',verbose=True)
    
    print("Done!")
    