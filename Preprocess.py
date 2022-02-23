import numpy as np
import argparse 
from os import walk
import os
import zivid
import shutil
from datetime import date
import time
import random


def readZividPCD(framefile,subsample):
    frame = zivid.frame.Frame(framefile)
    pcd = frame.point_cloud()

    if subsample is not None:
        pcd = pcd.downsample(subsample)
    
    xyz = pcd.copy_data("xyz")
    rgba = pcd.copy_data("rgba")
    normals = pcd.copy_data("normals")

    return xyz, rgba, normals


def saveZividPcdAsNpz(savedir,savename,framefile,normVectorSave = False, save_color=False, subsample = None):
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    xyz,rgba,normals = readZividPCD(framefile=framefile,subsample=subsample)

    if  normVectorSave and save_color:
        np.savez_compressed(os.path.join(savedir,savename),
                            rgb = rgba[:,:,0:3].reshape(np.shape(xyz)[0]*np.shape(xyz)[1],3),
                            pcd = xyz.reshape(np.shape(xyz)[0]*np.shape(xyz)[1],3),
                            normals = normals)
    
    elif  normVectorSave and not save_color:
        np.savez_compressed(os.path.join(savedir,savename),
                            pcd = xyz.reshape(np.shape(xyz)[0]*np.shape(xyz)[1],3),
                            normals = normals)

    elif  not normVectorSave and save_color:
        np.savez_compressed(os.path.join(savedir,savename),
                            rgb = rgba[:,:,0:3].reshape(np.shape(xyz)[0]*np.shape(xyz)[1],3),
                            pcd = xyz.reshape(np.shape(xyz)[0]*np.shape(xyz)[1],3))


    elif   not normVectorSave and not save_color:
        np.savez_compressed(os.path.join(savedir,savename),
                            pcd = xyz.reshape(np.shape(xyz)[0]*np.shape(xyz)[1],3))


def testZividSaveLoad():
    #TODO test saving and loading a zivid file usingsaveZividPcdAsNpz()
    FILEPATH = "C:\\Users\\jonas\\Documents\\NTNU\\MasterThesis\\DummyDataset\\Object1\\black cable - zvd1ps.zdf"
    saveZividPcdAsNpz('./testfolder','testname',FILEPATH)

    loaded = np.load('./testfolder/testname.npz')
    print(loaded["rgba"][200,200])
    print(loaded["xyz"][200,200])
    print(loaded["normals"][200,200])
    
def splitTrainVal(objectList,trainFile,valFile,valFraction = 0.1):

    trainN = int(len(objectList)*1-valFraction)

    trainList = random.sample(objectList,trainN)
    valList = []

    for n in objectList:
        if not trainList.__contains__(n):
            valList.append(n)


    f = open(trainFile,"a")
    for t in trainList:
        f.write(f"{t}\n")
    f.close()

    f = open(valFile,"a")
    for t in valList:
        f.write(f"{t}\n")
    f.close()


def main():
    #Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",help="Directory of Zivid pointclouds",type = str,default="C:/Users/jonas/Documents/NTNU/MasterThesis/DummyDataset")
    parser.add_argument("--zivid_camera_file",help="Path to .zfc camera for camera emulation", type=str, default="./FileCameraZividOne.zfc")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--include_normals",type=bool, default=False)
    parser.add_argument("--include_color",type= bool, default=False)
    parser.add_argument("--subsample", type=str, default = None, help="Default is no subsamplng, ommit this flag for raw data input. For subsampling input arguments by4x4 by3x3 or by2x2 ")
    parser.add_argument("--dataset_output_name",type=str,default="ZividDataset")
    parser.add_argument("--train_file_name",type= str, default="trainZivid.txt")
    parser.add_argument("--val_file_name",type= str, default="valZivid.txt")

    args = parser.parse_args()    

    # Connecting ZividFileCamera
    app = zivid.Application()
    camera = app.create_file_camera(args.zivid_camera_file)
    settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
    frame = camera.capture(settings)

    datasetFolder = args.dir

    print("\n")
    print("Preprocessing Zivid pointclouds")
    print("\n")
    print(f"=> Processing data in {datasetFolder}")

    directories = []

    for (dirpath, dirnames, filenames) in walk(datasetFolder):
        directories.extend(dirnames)

    if os.path.exists("./savefolder"):
        shutil.rmtree("./savefolder")
    os.makedirs("./savefolder")

    splitTrainVal(directories,os.path.join("./savefolder",args.train_file_name), os.path.join("./savefolder",args.val_file_name), args.val_fraction)

    for dir in directories:
        print(f"\nNow processing files in {os.path.join(datasetFolder,dir)}")
        files = os.listdir(os.path.join(datasetFolder,dir))
        i = 0
        f = open(os.path.join("./savefolder",str(dir)+"_all.txt"),"x")
        for file in files:
            print(file)
            saveZividPcdAsNpz("./saveFolder",f"{dir}_{i}",os.path.join(datasetFolder, dir, file),normVectorSave=args.include_normals, save_color=args.include_color,subsample=args.subsample)
            f.write(f"{dir}_{i}\n")
            i += 1
        f.close()
    #Create .txt files
    
    filesInSaveFolder = os.listdir("./savefolder")
    for file in filesInSaveFolder:
        if file.__contains__(".txt"):
            name = file.split("_all.")

            listFile = open(f"./savefolder/{file}","r")
            listFileContents = listFile.read().split("\n")

            pairFile = open(f"./savefolder/{name[0]}.txt","x")
            
            listFileContents1 = listFileContents
            usedElements = []
            for element1 in listFileContents1:
                for element2 in listFileContents1:
                    #print(element2)
                    if element1 == element2 or len(element1)<2 or len(element2)<2 or usedElements.__contains__(element1) or usedElements.__contains__(element2) :
                        None
                    else:
                        pairFile.write(f"{element1}.npz {element2}.npz\n")
                
                usedElements.append(element1)

            listFile.close()
            pairFile.close()

    t = time.localtime()
    current_time = time.strftime("%H%M", t)

    shutil.make_archive(f"{args.dataset_output_name}_{str(date.today())}_{str(current_time)}",'zip',"./saveFolder")
    shutil.rmtree("./saveFolder")





if __name__=="__main__":

   main()