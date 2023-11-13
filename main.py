# Imports

import torch
import sklearn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

''' Construção da Classe de Tokenização dos Dados '''
# Classe para tokenização dos dados
class TokenizaDados(Dataset):

    # Método construtor
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Método para calcular o comprimento do texto (cada sentença)
    def __len__(self):
        return len(self.texts)

    # Método para obter um item tokenizado
    def __getitem__(self, idx):

        # Obtém o índice do texto e do label
        text = self.texts[idx]
        label = self.labels[idx]

        # Aplica a tokenização
        inputs = self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }

if __name__ == '__main__':

    ''' Verificando o Ambiente de Desenvolvimento '''

    # Verifica se uma GPU está disponível e define o dispositivo apropriado

    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define o device (GPU ou CPU)
    device = torch.device(processing_device)
    print(device)

    ''' Funções Para os Loops de Treino, Avaliação e Previsão com Novos Dados '''

    # Método do loop de treino

    def treina_model(model, data_loader, criterion, optimizer, device):

        # Coloca o modelo em modo de treino
        model.train()

        # Inicializa o erro com zero
        total_loss = 0

        # Loop pelo data loader
        for batch in data_loader:

            # Extrai os ids do batch de dados e coloca no device
            input_ids = batch['input_ids'].to(device)

            # Extrai a máscara e coloca no device
            attention_mask = batch['attention_mask'].to(device)

            # Extrai os labels e coloca no device
            labels = batch['label'].to(device)

            # Zera os gradientes
            optimizer.zero_grad()

            # Faz as previsões
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Extrai o erro do modelo
            loss = outputs.loss

            # Aplica a otimização com backpropagation
            loss.backward()
            optimizer.step()

            # Acumula o erro
            total_loss += loss.item()

        return total_loss / len(data_loader)

    # Método do loop de avaliação
    def avalia_modelo(model, data_loader, criterion, device):

        model.eval()

        total_loss = 0

        with torch.no_grad():

            for batch in data_loader:

                input_ids = batch['input_ids'].to(device)

                attention_mask = batch['attention_mask'].to(device)

                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss

                total_loss += loss.item()

        return total_loss / len(data_loader)

    # Método do loop de previsão
    def predict(model, data_loader, device):

        model.eval()

        predictions = []

        with torch.no_grad():

            for batch in data_loader:

                input_ids = batch['input_ids'].to(device)

                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)

                _, preds = torch.max(outputs.logits, dim=1)

                predictions.extend(preds.tolist())

        return predictions

    ''' Definição dos Dados '''

    # Hiperparâmetros
    EPOCHS = 10
    BATCH_SIZE = 16
    MAX_LENGTH = 64
    LEARNING_RATE = 2e-5
    RANDOM_SEED = 42

    # Conjunto de dados de exemplo
    texts = [
        'A velocidade da luz é aproximadamente 300.000 km/s.',
        'A Terra é plana e os répteis controlam o mundo.',
        'A fotossíntese é um processo importante para as plantas.',
        'As vacas podem voar e atravessar paredes de concreto.',
        'O oxigênio é essencial para a respiração dos seres vivos.',
        'Os cavalos podem falar como seres humanos.',
        'As crianças aprendem a partir dos exemplos dos pais.',
        'As palavras verdadeiras não são agradáveis e as agradáveis não são verdadeiras.',
        'Leopardos trabalham de terno e gravata em frente ao computador.',
        'Carros voadores estão por toda parte.'
    ]

    labels = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  # 0: normal, 1: anômala

    # Divisão dos dados em treino e teste
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts,
                                                                          labels,
                                                                          test_size=0.2,
                                                                          random_state=RANDOM_SEED)

    ''' Tokenização dos Dados e Criação dos DataLoaders '''

    # Nome do modelo pré-treinado com 110M de parâmetros
    PRETRAINED_MODEL = 'bert-base-uncased'

    # Inicializa o tokenizador
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

    # Tokenização dos dados
    train_dataset = TokenizaDados(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = TokenizaDados(test_texts, test_labels, tokenizer, MAX_LENGTH)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    ''' Construção, Treinamento e Avaliação do Modelo '''

    # Importa o modelo pré-treinado
    modelo = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=2)

    # Coloca o modelo na memória do device
    modelo.to(device)

    # Configuração do otimizador e critério de perda
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Treinamento e validação do modelo
    for epoch in range(EPOCHS):

        train_loss = treina_model(modelo, train_loader, criterion, optimizer, device)

        test_loss = avalia_modelo(modelo, test_loader, criterion, device)

        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss}, Test Loss: {test_loss}')

    # Salvando o modelo
    torch.save(modelo, 'modelo.pt')

    ''' Deploy e Uso do Modelo Treinado '''

    # Teste de detecção de anomalias
    novos_dados = ['A gravidade mantém os planetas em órbita ao redor do Sol.',
                   'Os carros podem nadar no oceano como peixes.']

    # Tokeniza a amostra de dados
    novo_dataset = TokenizaDados(novos_dados, [0] * len(novos_dados), tokenizer, MAX_LENGTH)

    # Cria o dataloader
    novo_loader = DataLoader(novo_dataset, batch_size=BATCH_SIZE)

    # Faz as previsões com o modelo
    previsoes = predict(modelo, novo_loader, device)

    for text, prediction in zip(novos_dados, previsoes):
        print(f'Sentença: {text} | Previsão: {"anômala" if prediction else "normal"}')

