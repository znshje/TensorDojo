"""Export independent exercise templates without overwriting existing practice code."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import torch
torch.set_num_threads(1)
from catalog import PROBLEMS
if __name__=='__main__':
    folder=Path(__file__).resolve().parents[1]/'practice_templates'
    folder.mkdir(exist_ok=True)
    count=0
    for p in PROBLEMS.values():
        path=folder/(p['id']+'.py')
        if path.exists():continue
        path.write_text('# '+p['title']+'\n# '+p['description']+'\n# '+p['source']+'\n\n'+p['starter'])
        count+=1
    print(f'Exported {count} templates to {folder}; existing files preserved.')
