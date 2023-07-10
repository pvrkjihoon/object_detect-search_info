# 실시간 객체 정보 제공

실시간으로 객체를 인식하고, 객체의 정보를 검색해 송출하는것이 목표입니다.

## Requirement


```
* 10th generation Intel® CoreTM processor onwards
* At least 32GB RAM
* Ubuntu 22.04
* Python 3.9
```

## Clone code

* (Code clone 방법에 대해서 기술)

```shell
git clone https://github.com/pvrkjihoon/object_detect-search_info/blob/main/object_detect-search_info.py
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정방법에 대해 기술)

```shell
python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install wheel

python -m pip install openvino-dev

cd /path/to/repo/xxx/
python -m pip install -r requirements.txt

omz_downloader --name bert-small-uncased-whole-word-masking-squad-int8-0002 --precision FP16-INT8 --output_dir bert_model --cache_dir bert_model
```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

make
make install
```

## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

cd /path/to/repo/xxx/
python demo.py -i xxx -m yyy -d zzz
```

## Output

![./images/result.jpg](./images/result.jpg)

## Appendix

* (참고 자료 및 알아두어야할 사항들 기술)
