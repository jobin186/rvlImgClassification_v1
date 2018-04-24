#!/bin/bash
scp jobinkv@10.2.16.195:/var/www/cgi-bin/dv1/train.txt .
scp jobinkv@10.2.16.195:/var/www/cgi-bin/dv1/test.txt .
echo "Total No of training images are: " 
cat train.txt |wc -l
echo "No of testing images are: " 
cat test.txt |wc -l
