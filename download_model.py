import os
import gdown


os.makedirs('saves', exist_ok=True)
print('Downloading stcn.pth...')
gdown.download('https://drive.google.com/uc?id=1mRrE0uCI2ktdWlUgapJI_KmgeIiF2eOm', output='saves/stcn.pth', quiet=False)

print('Done.')