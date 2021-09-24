import torchvision
from torchvision import transforms
from pathlib import Path

#%%

# Run data generation
def generate(dataset, list_classes:list):
    labels = []
    for label_id in list_classes:
        labels.append(list(dataset.classes)[label_id])
    #print(labels)
    
    sub_dataset = []
    for datapoint in dataset:
        _, label_index = datapoint  # Extract label
        if label_index in list_classes:        
            sub_dataset.append(datapoint)    
    return sub_dataset


def test():
    '''load data and model'''
    project_dir = Path(__file__).resolve().parent.parent
    dataroot = project_dir / 'data'
    total_classes = 10 # [0-9]
    unlearn_class = 9
    
    mean=[125.31 / 255, 122.95 / 255, 113.87 / 255]
    std=[63.0 / 255, 62.09 / 255, 66.70 / 255]
    #transform_train = transforms.Compose([
    #        transforms.RandomCrop(32, padding=4),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        transforms.Normalize(mean, std),
    #        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
    #trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=False, transform=transform_test)
    
    print(len(testset))

    list_allclasses = list(range(total_classes))
    unlearn_listclass = [unlearn_class]
    list_allclasses.remove(unlearn_class) # rest classes
    #print(list_classes)
    unlearn_testdata = generate(testset, unlearn_listclass)
    rest_testdata = generate(testset, list_allclasses)
    print(len(unlearn_testdata), len(rest_testdata))

if __name__=='__main__':
    test()