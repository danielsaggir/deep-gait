import torch
import torch.nn as nn
import torch.nn.functional as F

def get_physical_adjacency_matrix(num_nodes=17):
    # רשימת החיבורים הפיזיים בגוף האדם (לפי 17 מפרקים)
    neighbor_links = [
        (0, 1), (0, 2), (1, 3), (2, 4), # פנים וראש
        (5, 6), (5, 7), (7, 9),         # כתפיים ויד שמאל
        (6, 8), (8, 10),                # יד ימין
        (11, 12), (5, 11), (6, 12),     # חיבורי גב ואגן
        (11, 13), (13, 15),             # רגל שמאל
        (12, 14), (14, 16)              # רגל ימין
    ]
    
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    # הוספת קשרים (גרף לא מכוון - החיבור עובד לשני הכיוונים)
    for i, j in neighbor_links:
        if i < num_nodes and j < num_nodes:
            A[i, j] = 1.0
            A[j, i] = 1.0
            
    # כל מפרק מחובר גם לעצמו (Self-loops)
    for i in range(num_nodes):
        A[i, i] = 1.0
        
    # נרמול שורות (כדי למנוע מ-loss להתפוצץ בגלל ערכים גבוהים)
    row_sum = A.sum(dim=1, keepdim=True)
    A = A / row_sum
    
    return A

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        A_init = get_physical_adjacency_matrix(num_nodes)
        self.A = nn.Parameter(A_init, requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_conv = self.conv(x)
        output = torch.einsum("nctv,vw->nctw", x_conv, self.A)
        return self.relu(self.bn(output))


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.conv(x)


class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, temporal_kernel=9, dropout=0.3):
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels, num_nodes)
        self.temporal = TemporalConv(
            out_channels,
            out_channels,
            temporal_kernel,
            padding=temporal_kernel // 2,
            dropout=dropout,
        )

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.spatial(x)

        n, c, t, v = x.shape
        x = x.permute(0, 3, 1, 2).reshape(n * v, c, t)
        x = self.temporal(x)
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        return x + res


class DeepGait_STGCN(nn.Module):
    def __init__(self, num_nodes=17, in_channels=11, embedding_dim=128):
        super().__init__()
        self.block1 = STGCN_Block(in_channels, 64, num_nodes)
        self.block2 = STGCN_Block(64, 128, num_nodes)
        self.block3 = STGCN_Block(128, 256, num_nodes)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class SiameseGaitVerifier(nn.Module):
    def __init__(self, num_nodes=17, in_channels=11, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = DeepGait_STGCN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward_once(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)

        abs_diff = torch.abs(z1 - z2)
        prod = z1 * z2
        cos = F.cosine_similarity(z1, z2, dim=1, eps=1e-8).unsqueeze(1)

        fused = torch.cat([abs_diff, prod, cos], dim=1)
        logits = self.classifier(fused)
        return logits, z1, z2
