# Segmentação tumoral HER2 end-to-end

O pipeline lê patches diretamente das lâminas, rasteriza os polígonos dos XMLs
Aperio em memória e treina uma U-Net/FPN/DeepLabV3+ com `Dice + Focal`. Nenhuma
imagem ou máscara intermediária é gravada em disco.

## Instalação e conferência

Além dos pacotes Python, o OpenSlide do sistema deve estar instalado.

```bash
pip install -r requirements.txt
python train.py --inspect
```

## Treino

```bash
python train.py \
  --data-root /mnt/HD_JULIO/Laminas_ROI/HER2-ROI \
  --architecture unet --encoder resnet50 \
  --channels 3 --patch-size 512 --batch-size 4 --epochs 30
```

Para RGB + Hematoxilina + DAB, use `--channels 5`. O encoder pré-treinado é
adaptado pelo `segmentation_models_pytorch`; os dois canais adicionais são obtidos
por deconvolução de cor em tempo de execução. A separação treino/validação é
feita antes da geração dos patches e, portanto, sempre por lâmina.

Os checkpoints e a configuração ficam em `runs/her2_unet/`. Ajuste
`--level` conforme a magnificação desejada (as coordenadas do XML são sempre
convertidas corretamente a partir do nível 0).

Cada execução também salva `metrics.csv` (loss e Dice por época) e `split.json`
(lâminas de treino e validação). Para GPUs pequenas, use `--batch-size 1
--freeze-bn`; se ocorrer instabilidade numérica, use `--no-amp` junto de patches
menores, por exemplo `--patch-size 256 --batch-size 2`.

## Inferência em patches

O comando usa automaticamente arquitetura, encoder e canais registrados no
checkpoint e salva patch, probabilidade, máscara, overlay e a matriz `.npy`:

```bash
python predict_patch.py --checkpoint runs/her2_unet_fp32_bs32/best.pt \
  --input patch.png --output inference/patch_01
```

Também é possível ler o patch diretamente de uma lâmina. `x` e `y` são
coordenadas no nível 0:

```bash
python predict_patch.py --checkpoint runs/her2_unet_fp32_bs32/best.pt \
  --slide lamina.svs --x 20000 --y 15000 --output inference/lamina_01
```

Para escolher automaticamente um patch dentro da maior região anotada, forneça
o XML no lugar das coordenadas: `--slide lamina.svs --xml anotacao.xml`.

## Avaliação de um split

Os treinos novos separam 70%/15%/15% por lâmina e registram treino, validação e
teste em `split.json`. A avaliação desenha o XML em azul, a predição em vermelho
e a interseção em magenta, além de salvar Dice e IoU:

```bash
python evaluate_split.py --checkpoint runs/her2_unet_fp32_bs32/best.pt \
  --split-json runs/her2_unet_fp32_bs32/split.json --split test \
  --output evaluation/test
```

Checkpoints anteriores, cujo `split.json` só contém treino e validação, podem
ser visualizados com `--split validation`; esse resultado não deve ser reportado
como teste independente.

Para obter uma única imagem panorâmica da lâmina, com XML azul e modelo
vermelho, use `evaluate_wsi.py`. A rede percorre a caixa das anotações com margem
na resolução usada no treino e projeta o resultado no thumbnail completo:

```bash
python evaluate_wsi.py --checkpoint runs/her2_unet_fp32/best.pt \
  --slide caso.svs --xml caso.xml --output evaluation/caso_overview.png
```

Uma lâmina também pode ser escolhida pelo índice no split: `--split-json
runs/her2_unet_fp32/split.json --split validation --index 0`.

Para gerar uma pasta por lâmina para todo o conjunto:

```bash
python evaluate_wsi_split.py --checkpoint runs/her2_unet_fp32/best.pt \
  --split-json runs/her2_unet_fp32/split.json --split validation \
  --output-root evaluation/validation
```

O processamento é retomável: casos que já possuem `overview.png` são ignorados.
