| Модель | Выбор | Сторона | LH / матчи | Deny / матчи | Смерти / матчи | До 120 с / матчи | (LH+deny) / циклы |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | sampling | radiant | 187/32 | 111/32 | 20/32 | 5/32 | 298/1883 |
| baseline | sampling | dire | 147/32 | 119/32 | 26/32 | 14/32 | 266/1402 |
| baseline | argmax | radiant | 142/32 | 84/32 | 24/32 | 14/32 | 226/926 |
| baseline | argmax | dire | 51/32 | 49/32 | 32/32 | 17/32 | 100/483 |
| honest | sampling | radiant | 44/32 | 26/32 | 32/32 | 18/32 | 70/356 |
| honest | sampling | dire | 315/32 | 136/32 | 5/32 | 0/32 | 451/2310 |
| honest | argmax | radiant | 0/32 | 0/32 | 32/32 | 32/32 | 0/0 |
| honest | argmax | dire | 2/32 | 0/32 | 2/32 | 0/32 | 2/21 |
| oracle | sampling | radiant | 225/32 | 133/32 | 23/32 | 4/32 | 358/1715 |
| oracle | sampling | dire | 199/32 | 132/32 | 29/32 | 15/32 | 331/1468 |
| oracle | argmax | radiant | 116/32 | 98/32 | 28/32 | 11/32 | 214/827 |
| oracle | argmax | dire | 0/32 | 0/32 | 32/32 | 32/32 | 0/0 |

| Модель | Сторона validation | ATTACK recall | ATTACK precision | Кадры |
| --- | --- | ---: | ---: | ---: |
| baseline | radiant | 6205/15274 | 6205/14279 | 659932 |
| baseline | dire | 7514/16543 | 7514/17334 | 659932 |
| honest | radiant | 2619/15274 | 2619/5747 | 659932 |
| honest | dire | 3588/16543 | 3588/7777 | 659932 |
| oracle | radiant | 10825/15274 | 10825/20549 | 659932 |
| oracle | dire | 12991/16543 | 12991/26662 | 659932 |
