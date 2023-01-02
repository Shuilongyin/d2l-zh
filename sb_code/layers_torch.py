import torch
from torch import nn


# ====================================================
# 如何初始化模型参数--Model
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        '''
        :初始化参数
        xavier初始化：nn.init.xavier_normal_(m.weight.data)
        kaiming初始化：nn.init.kaiming_normal_(m.weight.data)
        '''
        if isinstance(module, nn.Linear):  # linear层
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):  # embedding层
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):  # layernorm层
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]  #last_hidden_states: torch.Size([16, 133, 1024])
#         print('last_hidden_states:',last_hidden_states.shape)
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)  # weights: torch.Size([16, 133, 1])
#         print('weights:',weights.shape)
        feature = torch.sum(weights * last_hidden_states, dim=1)  #[16,1024] attention
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

