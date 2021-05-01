import smtplib
from email.mime.text import MIMEText
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import scipy.io as scio
from sklearn.datasets import load_digits


def send_message(subject:str, content:str):
    """
        send email from config.email_address_from to config.email_address_to
    """
    msg_from = '814499375@qq.com'   # 发送方邮箱地址。
    password = 'vfasazhqxmdgbeea'   # 发送方QQ邮箱授权码，不是QQ邮箱密码。
    msg_to = '814499375@qq.com'     # 收件人邮箱地址。

    msg = MIMEText(content, 'plain', 'utf-8')

    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to

    try:
        client = smtplib.SMTP_SSL('smtp.qq.com', smtplib.SMTP_SSL_PORT)
        print("连接到邮件服务器成功")

        client.login(msg_from, password)
        print("登录成功")

        client.sendmail(msg_from, msg_to, msg.as_string())
        print("发送成功")
    except smtplib.SMTPException as e:
        print("发送邮件异常")
    finally:
        client.quit()


def read_data(data_set):
    """
    get data from data set
    :return:
        X:{numpy array}, shape (n_samples, n_feature)
            data matrix
        Y:{numpy array}, shape (n_samples)
            true label
    """
    if data_set == 'MNIST':
        test_loader = get_test_dataloader()
        data = []
        label = []

        for (image, lab) in test_loader:
            data.append(image.numpy())
            label.append(lab.numpy())
        data = np.array(data)
        label = np.array(label)
        n, b, x, y, z = data.shape
        data = data.reshape(n, y*z)
        x, y = label.shape
        label = label.reshape(x*y)
    elif data_set == 'digits':
        dig = load_digits()
        data = dig.data
        label = dig.target
    else:
        mat = scio.loadmat('../data/' + data_set + '.mat')
        data = mat['X']
        data = data.astype(float)
        label = mat['Y']
        data = np.array(data)
        label = np.array(label)

    return data, label


def get_test_dataloader(batch_size=1, num_workers=0, shuffle=True):
    """ return training dataloader
    Args:
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(root='../data', train=False, download=False,
                                          transform=transform_test)
    test_loader = DataLoader(
        test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader

