import argparse
import math
import numpy as np
from util import *
from scipy import fftpack
from PIL import Image
from pprint import pprint

class JPEGFileReader:
    #header
    IMAGE_W = 32
    IMAGE_H = 32
    Q_N = 8
    Q_M = 8
    Q_SCALE = 8

    #data
    TABLE_SIZE_BITS = 16
    BLOCKS_COUNT_BITS = 32

    DC_CODE_LENGTH_BITS = 4
    CATEGORY_BITS = 4

    AC_CODE_LENGTH_BITS = 8
    RUN_LENGTH_BITS = 4
    SIZE_BITS = 4

    def __init__(self, filepath):
        # 생성자 메소드
        self.__file = open(filepath, 'r')

    def read_int(self, size):
        # bit to int로 변환 해주는 메소드
        if size == 0:
            return 0

        bin_num = self.__read_str(size)
        if bin_num[0] == '1':
            return self.__int2(bin_num)
        else:
            return self.__int2(binstr_flip(bin_num)) * -1

    def read_header(self):
        # txt파일 중 header를 읽어옴
        #각 할당된 bit수 만큼 읽어옴
        image_w = self.__read_uint(self.IMAGE_W)
        image_h = self.__read_uint(self.IMAGE_H)
        q_scale = self.__read_uint(self.Q_SCALE)
        q_m = self.__read_uint(self.Q_M)
        q_n = self.__read_uint(self.Q_N)

        return image_w, image_h, q_scale, q_m, q_n

    def read_dc_table(self):
        # dc 테이블을 읽어와 code table에 저장 하는 메소드
        #각 할당된 bit수 만큼 읽어옴
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            category = self.__read_uint(self.CATEGORY_BITS)
            code_length = self.__read_uint(self.DC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = category
        return table

    def read_ac_table(self):
        # ac 테이블을 읽어와 code table에 저장 하는 메소드
        # 각 할당된 bit수 만큼 읽어옴
        table = dict()

        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            run_length = self.__read_uint(self.RUN_LENGTH_BITS)
            size = self.__read_uint(self.SIZE_BITS)
            code_length = self.__read_uint(self.AC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = (run_length, size)
        return table

    def read_blocks_count(self):
        #block count를 읽어오는 메소드
        return self.__read_uint(self.BLOCKS_COUNT_BITS)

    def read_huffman_code(self, table):
        #huffman code를 읽어오는 메소드
        prefix = ''
        # TODO: break the loop if __read_char is not returing new char
        while prefix not in table:
            prefix += self.__read_char()
        return table[prefix]

    #bin to int or char나 str로 변환하는 메소드들
    def __read_uint(self, size):
        if size <= 0:
            raise ValueError("size of unsigned int should be greater than 0")
        return self.__int2(self.__read_str(size))

    def __read_str(self, length):
        return self.__file.read(length)

    def __read_char(self):
        return self.__read_str(1)

    def __int2(self, bin_num):
        return int(bin_num, 2)


def read_image_file(filepath):
    #txt파일을 읽어와 decoding 하는 메소드

    #객체 생성
    reader = JPEGFileReader(filepath)

    #이미지 header읽어옴
    image_w, image_h, q_scale, q_m, q_n = reader.read_header()

    print("header info")
    print(f'image w : %d' %image_w)
    print(f'image h : %d' %image_h)
    print(f'scale : %d' %q_scale)
    print(f'm :  %d' %q_m)
    print(f'n : %d' %q_n)

    #이미지 데이터 읽어옴
    tables = dict()
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
        if 'dc' in table_name:
            tables[table_name] = reader.read_dc_table()
            print(tables)
        else:
            tables[table_name] = reader.read_ac_table()
            #print(tables)


    blocks_count = reader.read_blocks_count()

    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    #decoding 수행 - block 수 만큼 반복
    for block_index in range(blocks_count):
        for component in range(3):
            dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if component == 0 else tables['ac_c']

            category = reader.read_huffman_code(dc_table)
            dc[block_index, component] = reader.read_int(category)

            cells_count = 0

            # TODO: try to make reading AC coefficients better
            while cells_count < 63:
                run_length, size = reader.read_huffman_code(ac_table)

                if (run_length, size) == (0, 0):
                    while cells_count < 63:
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                else:
                    for i in range(run_length):
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                    if size == 0:
                        ac[block_index, cells_count, component] = 0
                    else:
                        value = reader.read_int(size)
                        ac[block_index, cells_count, component] = value
                    cells_count += 1

    return image_w, image_h, q_scale, q_m, q_n, dc, ac, tables, blocks_count


def zigzag_to_block(zigzag):
    # assuming that the width and the height of the block are equal
    rows = cols = int(math.sqrt(len(zigzag)))

    if rows * cols != len(zigzag):
        raise ValueError("length of zigzag should be a perfect square")

    block = np.empty((rows, cols), np.int32)

    for i, point in enumerate(zigzag_points(rows, cols)):
        block[point] = zigzag[i]

    return block

def move(i, j):
    #양자화 테이블 만들기 위해 2d행렬을 형식에 맞게 변환하는 메소드
    #행렬값을 바꾸어 양자화 테이블의 값을 move
    if j < (8 - 1):
        return max(0, i - 1), j + 1
    else:
        return i + 1, j


def dequantize(block, scale, m, n):
    #dequantize 하는 메소드
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
        if i < int(m):
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

    return q, block * q


def idct_2d(image):
    #idct하는 메소드
    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


if __name__ == "__main__":
    print("------decoder------")

    #txt파일 읽어와 각 할당된 bit만큼 자른 후 각 인스턴스에 저장
    image_w, image_h, q_scale, q_m, q_n, dc, ac, tables, blocks_count = read_image_file("./result/encoder_output.txt")

    block_side = 8

    #이미지 당 한 라인에 블록의 갯수 저장
    blocks_per_line = image_h // block_side

    #이미지 사이즈 만큼 array생성
    npmat = np.empty((image_w, image_h, 3), dtype=np.uint8)

    #블록 갯수만큼 반복
    for block_index in range(blocks_count):
        #i - image_h / block
        i = block_index // blocks_per_line * block_side
        #j - image_w / block
        j = block_index % blocks_per_line * block_side

        #ycbcr만큼 반복
        for c in range(3):
            #zigzag scan하며 값을 가져옴
            zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
            quant_matrix = zigzag_to_block(zigzag)
            #idct수행
            q, dct_matrix = dequantize(quant_matrix, q_scale, q_m, q_n)
            block = idct_2d(dct_matrix)
            #decode한 값 저장
            npmat[i:i + 8, j:j + 8, c] = block + 128

    pprint(q)

    image = Image.fromarray(npmat, 'YCbCr')
    image = image.convert('RGB')
    image.save("./result/decode_output.bmp")
    image.save("./result/decode_output.jpg")