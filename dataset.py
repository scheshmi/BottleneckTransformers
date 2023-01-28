from typing import Tuple
from torch.utils.data import  DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from config import BATCH_SIZE

def get_dataset(train_dir, test_dir)-> Tuple[DataLoader]:

    # intel image classification dataset
    

    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        #  transforms.RandomRotation(30),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4302, 0.4575, 0.4538],[0.2606, 0.2588, 0.2907],inplace = True)])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4302, 0.4575, 0.4538],[0.2606, 0.2588, 0.2907],inplace = True)])


    train_ds = datasets.ImageFolder(root=train_dir,transform=train_transforms)
    test_ds = datasets.ImageFolder(root=test_dir,transform=test_transforms)

    train_loader = DataLoader(dataset = train_ds,batch_size = BATCH_SIZE,shuffle = True,num_workers=2,pin_memory=True,drop_last=True)

    test_loader = DataLoader(dataset = test_ds,batch_size = BATCH_SIZE,num_workers=2, pin_memory=True,drop_last=True)

    return train_loader, test_loader
