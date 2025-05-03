# GPU Bitcoin Private Key Finder

Este programa utiliza CUDA para realizar busca paralela de chaves privadas Bitcoin que correspondam a um endereço específico. O programa permite especificar índices específicos da chave privada para realizar permutações, mantendo os demais dígitos fixos.

## Requisitos

- NVIDIA GPU com suporte a CUDA
- CUDA Toolkit instalado
- Biblioteca secp256k1
- Sistema operacional compatível com CUDA

## Compilação

```bash
make
```

## Uso

```bash
./cacachavecuda -c <chave_privada> -e <endereco_bitcoin> -i <indices> [-block <tamanho>] [-grid <tamanho>]
```

Onde:
- `-c`: Chave privada em formato hexadecimal (64 dígitos)
- `-e`: Endereço Bitcoin alvo em formato hash160 (40 dígitos hexadecimais)
- `-i`: Índices a serem permutados, separados por vírgula (1-64)
- `-block`: Tamanho do bloco CUDA (opcional, padrão: 256)
- `-grid`: Tamanho do grid CUDA (opcional, padrão: 4096)

### Exemplo

```bash
./cacachavecuda -c 000000000000000000000000000000000000000000000000000000000000000000 -e 20d45a6a76253570c0e9e0b216e319943335db8a5 -i 1,2,3,4
```

Este comando irá:
1. Usar a chave privada fornecida como base
2. Tentar todas as permutações possíveis nos índices 1, 2, 3 e 4
3. Procurar por uma chave que gere o endereço Bitcoin alvo
4. Se encontrar, salvará a chave em ACHADOS.txt

## Limitações

- Máximo de 12 índices para permutação
- Os índices devem estar entre 1 e 64
- A chave privada deve ser válida (64 dígitos hexadecimais)
- O endereço Bitcoin deve estar no formato hash160 (40 dígitos hexadecimais)

## Saída

O programa exibirá:
- Número total de permutações a serem testadas
- Velocidade atual de processamento
- Total de chaves testadas
- Tempo de execução

Se uma chave correspondente for encontrada, ela será:
- Exibida no console
- Salva no arquivo ACHADOS.txt