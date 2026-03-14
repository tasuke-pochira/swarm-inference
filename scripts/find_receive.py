from pathlib import Path

p = Path(r"X:\projects\GIT-LOKI\130326\swarm_inference\src\network.rs")
text = p.read_text(encoding='utf-8')
count = text.count('pub async fn receive_message')
print('count', count)
for i, line in enumerate(text.splitlines(), 1):
    if 'pub async fn receive_message' in line:
        print(i, line)
