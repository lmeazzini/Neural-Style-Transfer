import numpy as np
from PIL import Image
import sys
from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
import glob
import os
from scipy.optimize import fmin_l_bfgs_b

media_rgb_imagenet = [123.68, 116.779, 103.939] # Média dos valores RGB das imagens do ImageNet
peso_conteudo = 0.02
peso_estilo = 4.5
variacao_peso = 0.995
variacao_fator_custo = 1.25
largura_imagem = 260
altura_imagem = 260
canais = 3 # R G B

if len(sys.argv) == 1:
    path_entrada = 'Imagens/win_xp.jpg'
    path_estilo = 'Imagens/barnes.jpg'
else:
    path_entrada = str(sys.argv[1])
    path_estilo = str(sys.argv[2])

img_entrada = Image.open(path_entrada)
img_entrada = img_entrada.resize((altura_imagem,largura_imagem))


img_estilo = Image.open(path_estilo)
img_estilo = img_estilo.resize((altura_imagem,largura_imagem))


img_entrada_arr = np.asarray(img_entrada, dtype="float32") # shape = (largura_imagem, altura_imagem, canais)
img_entrada_arr = np.expand_dims(img_entrada_arr, axis=0) # shape = (1, largura_imagem, altura_imagem, canais)
img_entrada_arr[:, :, :, 0] -= media_rgb_imagenet[2]
img_entrada_arr[:, :, :, 1] -= media_rgb_imagenet[1]
img_entrada_arr[:, :, :, 2] -= media_rgb_imagenet[0]
img_entrada_arr = img_entrada_arr[:, :, :, ::-1] # Troca RGB por BGR

img_estilo_arr = np.asarray(img_estilo, dtype="float32") # shape = (largura_imagem, altura_imagem, canais)
img_estilo_arr = np.expand_dims(img_estilo_arr, axis=0) # shape = (1, largura_imagem, altura_imagem, canais)
img_estilo_arr[:, :, :, 0] -= media_rgb_imagenet[2]
img_estilo_arr[:, :, :, 1] -= media_rgb_imagenet[1]
img_estilo_arr[:, :, :, 2] -= media_rgb_imagenet[0]
img_estilo_arr = img_estilo_arr[:, :, :, ::-1] # Troca RGB por BGR


entrada = K.variable(img_entrada_arr)
estilo = K.variable(img_estilo_arr)
imagem_combinada = K.placeholder((1, largura_imagem, altura_imagem, canais))

tensor_entrada = K.concatenate([entrada, estilo, imagem_combinada], axis=0)
model = VGG16(input_tensor=tensor_entrada, include_top=False, weights='imagenet')


def custo_conteudo(conteudo, combinacao):
    return K.sum(K.square(combinacao - conteudo))

layers = dict([(layer.name, layer.output) for layer in model.layers])

camada_conteudo = 'block2_conv2' #Usando a camada após a primeiro convolução os resultados são melhores
camada_caracteristicas = layers[camada_conteudo]
camada_conteudo_caracteristicas = camada_caracteristicas[0, :, :, :]
caracteristicas_combinacao = camada_caracteristicas[2, :, :, :]

custo = K.variable(0.)
custo += peso_conteudo * custo_conteudo(camada_conteudo_caracteristicas, caracteristicas_combinacao)


def gram_matrix(x):
    caracteristicas = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(caracteristicas, K.transpose(caracteristicas))
    return gram

def calc_custo_estilo(estilo, combincacao):
    estilo = gram_matrix(estilo)
    combincacao = gram_matrix(combincacao)
    tamanho = largura_imagem * altura_imagem
    return K.sum(K.square(estilo - combincacao)) / (4. * (canais ** 2) * (tamanho ** 2))

camadas_estilo = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
for nome in camadas_estilo:
    caracteristicas_camada = layers[nome]
    caracteristicas_estilo = caracteristicas_camada[1, :, :, :]
    caracteristicas_combinacao = caracteristicas_camada[2, :, :, :]
    custo_estilo = calc_custo_estilo(caracteristicas_estilo, caracteristicas_combinacao)
    custo += (peso_estilo / len(camadas_estilo)) * custo_estilo


def custo_variacao_total(x):
    a = K.square(x[:, :largura_imagem-1, :altura_imagem-1, :] - x[:, 1:, :altura_imagem-1, :])
    b = K.square(x[:, :largura_imagem-1, :altura_imagem-1, :] - x[:, :altura_imagem-1, 1:, :])
    return K.sum(K.pow(a + b, variacao_fator_custo))

custo += variacao_peso * custo_variacao_total(imagem_combinada)


saidas = [custo]
saidas += K.gradients(custo, imagem_combinada)

def calculo_custo_e_gradientes(x):
    x = x.reshape((1, largura_imagem, altura_imagem, canais))
    outs = K.function([imagem_combinada], saidas)([x])
    custo = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return custo, gradients

class Evaluator:

    def custo(self, x):
        custo, gradientes = calculo_custo_e_gradientes(x)
        self._gradientes = gradientes
        return custo

    def gradientes(self, x):
        return self._gradientes

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, largura_imagem, altura_imagem, canais)) - 128. #iniciação aleatória
n = 10 # numero de iteracoes
for i in range(n):
    x, custo, info = fmin_l_bfgs_b(evaluator.custo, x.flatten(), fprime=evaluator.gradientes, maxfun=20)
    print("Iteracao %d completa com custo: %d" % (i + 1, custo))
    
x = x.reshape((largura_imagem, altura_imagem, canais))
x = x[:, :, ::-1] # BGR para RGB
# Retira a normalização pela média da ImageNet
x[:, :, 0] += media_rgb_imagenet[2]
x[:, :, 1] += media_rgb_imagenet[1]
x[:, :, 2] += media_rgb_imagenet[0]

x = np.clip(x, 0, 255).astype("uint8") # mantem os valores entre 0 e 255
output_image = Image.fromarray(x)
output_image.save('output.png')


combinada = Image.new("RGB", (largura_imagem*3, altura_imagem))
x_offset = 0
for image in map(Image.open, ['entrada.png', 'estilo.png', 'output.png']):
    combinada.paste(image, (x_offset, 0))
    x_offset += largura_imagem
combinada.save('vis.png')

