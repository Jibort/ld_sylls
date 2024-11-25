# ld_sylls - Separador de Síl·labes amb Xarxes Neuronals

Aquest projecte implementa un separador de síl·labes en català utilitzant xarxes neuronals. El projecte està desenvolupat en Go i suporta tant execució en CPU com acceleració per GPU utilitzant CUDA.

## Estat del Projecte

Actualment en desenvolupament actiu. Implementat:
- [x] Tipus de dades FixedPoint per a càlculs precisos
- [x] Estructura bàsica de la xarxa neuronal
- [x] Forward propagation
- [x] Backpropagation
- [ ] Entrenament amb dades reals
- [ ] Acceleració CUDA
- [ ] Interfície d'usuari

## Requisits

- Go 1.21 o superior
- Per suport CUDA (opcional):
  - NVIDIA CUDA Toolkit 12.0 o superior
  - Targeta gràfica NVIDIA compatible amb CUDA

## Instal·lació

```bash
# Clonar el repositori
git clone https://github.com/Jibort/ld_sylls.git

# Entrar al directori
cd ld_sylls

# Compilar
go build ./cmd/ld_sylls
```

## Ús

[Pendent de documentar]

## Tecnologies Utilitzades

- Go 1.21
- CUDA (planificat)
- Fixed-point arithmetic per a càlculs precisos
- Xarxes neuronals feed-forward amb backpropagation

## Llicència

[Pendent de definir]

## Autor

Joan Ignat Quintana (JIQ)