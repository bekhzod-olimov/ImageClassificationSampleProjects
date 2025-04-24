from torchvision import transforms as T

def get_tfs(im_size=224, mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]): 

    return T.Compose([
            T.Resize((im_size, im_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
            ])