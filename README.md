## Documentation

- [Install requirements](#install-requirements)
- [How to use script](#how-to-use-script)
- [Failure attempts](#failure-attempts)

---
### Install requirements 

You can check this commands in "HowToUse" notebook. Write these commands in command line in their order.

- ```apt-get install python-opencv```

- ``` pip install solaris  ```

- ```apt-get install -qq curl g++ make ```

- ```curl -L http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz | tar xz ```

- ```
  import os
  os.chdir('spatialindex-src-1.8.5')
  ```

- ```
  ./configure;
  ```

- ```make ```

- ```make install```

- ```pip install rtree ```

### How to use notebook and script (api.py)

###  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BW9XHY59CQcUr7Ya8M2RpHdcO4fi-5Xg#scrollTo=VaCeGwEizhlA)

- Open notebook in google colab 
- Then upload files: "weights/xdxd_spacenet4_solaris_weights.pth" and "notebook/img.png"
- Run all cells

- In this notebook you can see how to install requirements (tested only in google colab with linux system). And also what is going on under the script and how to use this scripts. 
- In my example I use png/jpeg format images. But If you want to use other formats, drop me a line about it, I will code custom function for you.
- Also note output is (512, 512) size, therefore I made function **save_img_with_mask** with **output_size **param, you can your output size. If **output_size=None**, then **output_size=input_img_size**

### Failure attempts

- https://github.com/ternaus/TernausNetV2 - problems with installation, I found the same problem in repo Issues, but was no answers there.
- https://github.com/azavea/raster-vision - good framework, however I am not familiar with docker right now + it so hard to use it in simple python script
- https://github.com/YuansongFeng/satellite_building_detection.pytorch - does not have pretrained weights



