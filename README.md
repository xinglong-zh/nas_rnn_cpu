###  pytorch project fot NAS  
---
+ cifar-cnn 
        
        docker build ./ -t "name:tag"            // build image
        docker run -p 19010:19010 name:tag        //正常运行
        docker run -d -p 19010:19010 name:tag     //后台执行
