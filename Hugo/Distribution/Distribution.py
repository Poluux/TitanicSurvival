import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier
df = pd.read_csv('Project 2_Titanic-Dataset.csv')

# Distribution de SibSp
sibsp_distribution = df['SibSp'].value_counts().sort_index()

# Distribution de Parch
parch_distribution = df['Parch'].value_counts().sort_index()

# Distribution de Embarked
embarked_distribution = df['Embarked'].value_counts().sort_index()

# Affichage côte à côte sur 3 colonnes
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# SibSp
sibsp_distribution.plot(kind='bar', ax=axes[0], color='lightcoral')
axes[0].set_title("Frères/Sœurs ou conjoint·e (SibSp)")
axes[0].set_xlabel("Nombre de proches")
axes[0].set_ylabel("Nombre de passagers")
axes[0].set_xticks(sibsp_distribution.index)

# Parch
parch_distribution.plot(kind='bar', ax=axes[1], color='mediumseagreen')
axes[1].set_title("Parents/enfants (Parch)")
axes[1].set_xlabel("Nombre de proches")
axes[1].set_ylabel("Nombre de passagers")
axes[1].set_xticks(parch_distribution.index)

# Embarked
embarked_distribution.plot(kind='bar', ax=axes[2], color='skyblue')
axes[2].set_title("Port d'embarquement (Embarked)")
axes[2].set_xlabel("Port")
axes[2].set_ylabel("Nombre de passagers")
axes[2].set_xticks(range(len(embarked_distribution.index)))
axes[2].set_xticklabels(embarked_distribution.index)

plt.tight_layout()
plt.show()
