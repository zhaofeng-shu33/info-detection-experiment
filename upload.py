import os
import oss2

if __name__ == '__main__':
    os.chdir('build')
    access_key_id = os.getenv('AccessKeyId')
    access_key_secret = os.getenv('AccessKeySecret')
    if(access_key_secret is not None):
        auth = oss2.Auth(access_key_id, access_key_secret)
        bucket = oss2.Bucket(auth, 'http://oss-cn-shenzhen.aliyuncs.com', 'programmierung')
        research_base = 'research/info-clustering/experiment/outlier_detection/'                
        for i in os.listdir('./'):
            file = open(i, 'rb')
            bucket.put_object(research_base + i, file)
           
