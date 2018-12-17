import argparse
import os
import math
import numpy as np
from util import *
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree
from pprint import pprint

def zero_padding(h, w, numpy_image):
    # 이미지크기가 8로 나누어 떨러지지 않을때 zeropadding하는 메소드
    # 이미지 크기가 홀수 일때는 아래, 오른쪽으로 +1 padding됨
    # w,h -> image size
    # numpy_image -> padding할 numpy image
    #상 하 padding
    if h % 8 != 0:
        pad_n = int((8 - int((h % 8))) / 2)
        #print(pad_n)
        if w % 2 == 0:
            numpy_image = np.pad(numpy_image, ((pad_n, pad_n), (0, 0), (0, 0)), 'constant')
        else:
            numpy_image = np.pad(numpy_image, ((pad_n, pad_n + 1), (0, 0), (0, 0)), 'constant')

    #좌 우 padding
    if w % 8 != 0:
        pad_n = int((8 - int((w % 8))) / 2)
        #print(pad_n)

        if h % 2 == 0:
            numpy_image = np.pad(numpy_image, ((0, 0), (pad_n, pad_n), (0, 0)), 'constant')
        else:
            numpy_image = np.pad(numpy_image, ((0, 0), (pad_n, pad_n + 1), (0, 0)), 'constant')

    #print(numpy_image.shape)
    return numpy_image

def move(i, j):
    #양자화 테이블 만들기 위해 2d행렬을 형식에 맞게 변환하는 메소드
    #행렬값을 바꾸어 양자화 테이블의 값을 move
    if j < (8 - 1):
        return max(0, i - 1), j + 1
    else:
        return i + 1, j


def quantize(block, scale, m, n):
    #양자화 하는 메소드
    #8*8 0값의 matrix 생성

    q = [[0] * 8 for _ in range(8)]

    #input값을 기반으로 1d tmp list만듬
    #0번째는 16
    #1~m -> scale
    #m+1부터는 -> scale*n
    tmp = []
    tmp.append(16)               #0번째는 16

    #반복문 돌며 1~m -> scale, m+1부터는 -> scale*n 수행
    for i in range(1, 64):
        if i <= int(m):
            tmp.append(int(scale))
        else:
            tmp.append(int(n) * int(scale))

    #양자화 block make
    x, y = 0, 0
    for v in range(64):
        q[y][x] = tmp[v]
        if (x + y) & 1:
            x, y = move(x, y)
        else:
            y, x = move(y, x)
    return q, (block / q).round().astype(np.int32)

def block_to_zigzag(block):
    # 지그재그 스캔하며 array make하는 메소드
    # zigzag_points는 util메소드 참조
    return np.array([block[point] for point in zigzag_points(*block.shape)])


def dct_2d(image):
    # dct수행하는 메소드
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def run_length_encode(arr):
    # run-length encoding 수행하는 메소드
    # sss와 value의 형태로 부호화

    #sequence가 끝나는 곳을 결정
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # symbols = (sss, value)로 구성
    symbols = []

    # value에 대한 코드
    values = []

    run_length = 0
    # 반복문 돌며 run-length 수행
    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values


def write_to_file(filepath, rows, cols, scale, m, n, dc, ac, blocks_count, tables):
    # 지금까지 구한 값들을 모두 binary로 바꾸어 txt file generate하는 메소드
    try:
        f = open(filepath, 'w')
    except FileNotFoundError as e:
        raise FileNotFoundError(
                "No such directory: {}".format(
                    os.path.dirname(filepath))) from e

    # header 구성
    f.write(uint_to_binstr(rows, 32))
    f.write(uint_to_binstr(cols, 32))
    f.write(uint_to_binstr(int(scale), 8))
    f.write(uint_to_binstr(int(m), 8))
    f.write(uint_to_binstr(int(n), 8))

    # image encoding  data 구성
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        #table size는 16bit
        f.write(uint_to_binstr(len(tables[table_name]), 16))

        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                # category는 4bit
                # dc code length는 4 bit
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)
            else:
                #run-length는 4bit
                #size는 4bit
                #ac code length는 8bit
                f.write(uint_to_binstr(key[0], 4))
                f.write(uint_to_binstr(key[1], 4))
                f.write(uint_to_binstr(len(value), 8))
                f.write(value)

    # block count 는 32bit
    f.write(uint_to_binstr(blocks_count, 32))

    for b in range(blocks_count):
        for c in range(3):
            category = bits_required(dc[b, c])
            symbols, values = run_length_encode(ac[b, :, c])

            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']

            f.write(dc_table[category])
            f.write(int_to_binstr(dc[b, c]))

            for i in range(len(symbols)):
                f.write(ac_table[tuple(symbols[i])])
                f.write(values[i])
    f.close()

#=========================
#메인
#=========================
if __name__ == "__main__":
    print("------encoder------")
    # 양자화 테이블 생성시 필요 인스턴스 input 받음
    scale = input("scale : ")
    m = input("m : ")
    n = input("n : ")

    # 이미지 불러옴
    input_file = "./image/test.bmp"          # bmp image 경로
    output_file = "./result/encoder_output.txt"        #output file 경로

    # =========================
    # 전처리 과정
    # =========================
    image = Image.open(input_file)
    ycbcr = image.convert('YCbCr')          # ycbcr로 변환

    npmat = np.array(ycbcr, dtype=np.uint8)  # numpy형식으로 바꿈

    # image size
    rows, cols = npmat.shape[0], npmat.shape[1]
    print(f'%d, %d' %(cols, rows))

    # 상하, 좌우 zeropadding 수행(image size % 8 !=0)
    n_image = zero_padding(rows, cols, npmat)
    rows, cols = n_image.shape[0], n_image.shape[1]
    print(f'zero padding after (%d, %d)' % (cols,rows))

    blocks_count = rows // 8 * cols // 8        # block size: 8x8

    # =========================
    # 양자화 과정
    # =========================
    # dc 1개, ac 63개 블록 생성
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)


    # 블록 갯수만큼 반복
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                block_index += 1
            except NameError:
                block_index = 0

            # Y, Cb,Cr 만큼 반복
            for k in range(3):
                # level shift수행
                block = n_image[i:i + 8, j:j + 8, k] - 128

                dct_matrix = dct_2d(block)
                q, quant_matrix = quantize(dct_matrix, scale, m, n)

                # =========================
                # 부호화
                # =========================
                zz = block_to_zigzag(quant_matrix)

                dc[block_index, k] = zz[0]
                ac[block_index, :, k] = zz[1:]

    pprint(q)

    # =========================
    # 허프만 부호화
    # =========================
    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    H_AC_Y = HuffmanTree(
        flatten(run_length_encode(ac[i, :, 0])[0]
                for i in range(blocks_count)))
    H_AC_C = HuffmanTree(
        flatten(run_length_encode(ac[i, :, j])[0]
                for i in range(blocks_count) for j in [1, 2]))


    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}

    print('dc_y')
    print(tables['dc_y'])
    print('ac_y')
    print(tables['ac_y'])
    print('dc_c')
    print(tables['dc_c'])
    print('ac_c')
    print(tables['ac_c'])


    # =========================
    # 프레임 빌더
    # =========================
    write_to_file(output_file,cols, rows, scale, m, n, dc, ac, blocks_count, tables)