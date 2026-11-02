"""Masked imitation metrics. Conditional head scores are explicitly named."""
import html
import json
from pathlib import Path

import torch
from .spec import Act, COORDINATES


class ImitationMetrics:
    def __init__(self):
        self.totals = {}
        self.confusion = None

    @torch.no_grad()
    def update(self, predicted, labels, valid):
        a, p, item = labels['action'], predicted['action'], labels['item']
        equal = {k: predicted[k] == labels[k] for k in labels}
        moving = (a == Act.MOVE) | (a == Act.MOVE_DIRECT)
        attacking = a == Act.ATTACK
        razing = (a >= Act.RAZE_NEAR) & (a <= Act.RAZE_FAR)
        spatial = moving | ((a == Act.ITEM) & (item == 3))
        targeted = attacking | ((a == Act.ITEM) & ((item == 3) | (item == 6)))
        command = (equal['action'] & ((a != Act.ITEM) | equal['item'])
                   & (~targeted | equal['target']) & (~spatial | (equal['x'] & equal['y'])))
        def add(name, values, mask):
            mask = mask & valid
            pair = torch.stack(((values * mask).sum(), mask.sum())).double()
            self.totals[name] = self.totals.get(name, torch.zeros_like(pair)) + pair
        add('action_accuracy', equal['action'], valid)
        add('active_action_accuracy', equal['action'], a != Act.CONTINUE)
        add('full_command_accuracy', command & equal['buy'], valid)
        add('active_full_command_accuracy', command & equal['buy'], (a != Act.CONTINUE) | (labels['buy'] != 0))
        for name, mask in [('move', moving), ('attack', attacking), ('raze', razing)]:
            add(name + '_recall', equal['action'], mask)
            predicted_mask = ((p == Act.MOVE) | (p == Act.MOVE_DIRECT)) if name == 'move' else (p == Act.ATTACK) if name == 'attack' else ((p >= Act.RAZE_NEAR) & (p <= Act.RAZE_FAR))
            add(name + '_precision', equal['action'], predicted_mask)
        add('attack_with_target_recall', equal['action'] & equal['target'], attacking)
        add('target_accuracy_given_teacher_action', equal['target'], targeted)
        add('item_accuracy_given_teacher_action', equal['item'], a == Act.ITEM)
        add('buy_accuracy', equal['buy'], valid)
        add('purchase_recall', equal['buy'], labels['buy'] != 0)
        add('purchase_precision', equal['buy'], predicted['buy'] != 0)
        spacing = COORDINATES[1] - COORDINATES[0]
        distance = torch.sqrt((predicted['x'] - labels['x']).float().square() + (predicted['y'] - labels['y']).float().square()) * spacing
        add('move_distance_mean_given_teacher_action', distance, moving)
        add('move_within_64_units', equal['action'] & (distance <= 64), moving)
        add('move_within_128_units', equal['action'] & (distance <= 128), moving)
        matrix = torch.bincount((a[valid] * len(Act) + p[valid]), minlength=len(Act)**2).reshape(len(Act), len(Act))
        self.confusion = matrix if self.confusion is None else self.confusion + matrix

    def summary(self):
        result = {}
        if self.totals:
            pairs = torch.stack(list(self.totals.values())).cpu().tolist()
            for key, (total, count) in zip(self.totals, pairs):
                result[key] = {'value': total / count if count else None, 'count': int(count)}
        if self.confusion is not None:
            matrix = self.confusion.cpu().tolist()
            classes = {}
            for action in Act:
                actual = sum(matrix[action]); predicted = sum(row[action] for row in matrix)
                classes[action.name.lower()] = {'count': actual, 'predicted_count': predicted, 'recall': matrix[action][action] / actual if actual else None, 'precision': matrix[action][action] / predicted if predicted else None}
            recalls = [r['recall'] for r in classes.values() if r['count']]
            result['balanced_action_recall'] = {'value': sum(recalls) / len(recalls) if recalls else None, 'count': len(recalls)}
            result['per_action'] = classes
            result['confusion_matrix'] = matrix
        return result


def write_progress(out, progress, history):
    """A self-refreshing local report; no server or additional dependencies."""
    out = Path(out)
    temporary = out / 'status.tmp'
    temporary.write_text(json.dumps(progress, indent=2), encoding='utf-8')
    temporary.replace(out / 'status.json')
    columns = [('Текущий проход', progress.get('metrics', {}))]
    if history:
        columns += [('Последняя валидация', history[-1]['validation']['metrics'])]
    names = {
        'action_accuracy': 'Совпадение типа действия',
        'balanced_action_recall': 'Среднее совпадение по типам действий',
        'active_full_command_accuracy': 'Полная активная команда, включая параметры и покупки',
        'attack_with_target_recall': 'Атака: совпали команда и цель',
        'attack_precision': 'Точность предсказанных атак',
        'raze_recall': 'Доля воспроизведённых койлов (нужный койл)',
        'raze_precision': 'Точность предсказанных койлов',
        'move_recall': 'Доля воспроизведённых команд движения',
        'move_distance_mean_given_teacher_action': 'Ошибка точки движения, игровые единицы *',
        'move_within_128_units': 'Движение: нужная команда и точка в пределах 128 единиц',
        'purchase_recall': 'Доля воспроизведённых покупок',
        'purchase_precision': 'Точность предсказанных покупок',
    }
    rows = []
    for key, title in names.items():
        cells = []
        for _, values in columns:
            entry = values.get(key, {}); value = entry.get('value')
            label = '—' if value is None else f'{value:.1f}' if 'distance' in key else f'{value:.1%}'
            cells.append(f'<td>{label} <small>(n={entry.get("count", 0)})</small></td>')
        rows.append('<tr><th>' + title + '</th>' + ''.join(cells) + '</tr>')
    epochs = ''.join(f'<tr><td>{r["epoch"]}</td><td>{r["train"]["loss"]:.4f}</td><td>{r["validation"]["loss"]:.4f}</td></tr>' for r in history)
    document = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>SF — обучение</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;background:#111827;color:#e5e7eb}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #374151}}small{{color:#9ca3af}}h1{{font-size:26px}}</style>
<h1>SF — обучение повторению учителя</h1><p>Эпоха {progress.get('epoch', 0)}, {html.escape(progress.get('split', ''))}. Шагов оптимизатора: {progress.get('optimizer_steps', 0)}. Минут: {progress.get('minutes', 0)}. Loss: {progress.get('loss', 0):.4f} (меньше — лучше).</p>
<table><tr><th>Метрика</th>{''.join('<th>'+name+'</th>' for name,_ in columns)}</tr>{''.join(rows)}</table>
<p>* Ошибка координат вычисляется при заданном типе действия учителя. Полная команда и движение в пределах 128 единиц дополнительно требуют правильного типа действия. Высокая общая accuracy может скрывать редкие ошибки на койлах и атаках.</p>
<table><tr><th>Эпоха</th><th>Train loss</th><th>Validation loss</th></tr>{epochs}</table>
<p>Окончательный результат определяет также проверка в Dota: выход на мид, атаки крипов и применение койлов. Страница обновляется каждые 15 секунд; данные — каждые 25 батчей и в конце эпохи.</p></html>'''
    temporary = out / 'metrics.tmp'
    temporary.write_text(document, encoding='utf-8'); temporary.replace(out / 'metrics.html')
