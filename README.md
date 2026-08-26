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
