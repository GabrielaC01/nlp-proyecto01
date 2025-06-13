import torch
import torch.nn.functional as F
import math

class GeneradorTexto:
    def __init__(self, modelo, vocab_inv):
        self.modelo = modelo
        self.vocab_inv = vocab_inv

    def generar_greedy(self, entrada, num_tokens=10):
        self.modelo.eval()
        salida = entrada.clone()

        for _ in range(num_tokens):
            mask = self.modelo.generate_causal_mask(salida.size(1)).to(salida.device)

            with torch.no_grad():
                emb = self.modelo.embedding(salida) * math.sqrt(self.modelo.d_model)
                emb = self.modelo.pos_encoder(emb)
                emb = emb.transpose(0, 1)
                logits = self.modelo.transformer_decoder(emb, emb, tgt_mask=mask)
                logits = self.modelo.fc_out(logits)

            ultimo_logit = logits[-1, 0, :]
            siguiente_token = torch.argmax(F.softmax(ultimo_logit, dim=-1), dim=-1)
            siguiente_token = siguiente_token.unsqueeze(0).unsqueeze(0)
            salida = torch.cat([salida, siguiente_token], dim=1)

        return "".join([self.vocab_inv[idx.item()] for idx in salida[0]])

