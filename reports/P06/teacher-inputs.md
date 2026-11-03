# Teacher-v9: входы решения ATTACK / LH / deny

Источник: `gpu/src/scripted_teacher.rs`, `decide`, `own_hit`, `physical`, `flight` на исходном `5ea5a43`.
Учитель не читает RNG, будущие состояния или уже выпавший урон снаряда. Ниже перечислены также условия, которые подавляют ATTACK.
Св = свой SF, Вр = видимый вражеский SF, К = видимые мили/рейндж-крипы, Б = видимые башни.
Числа вызовов и ошибок: `api-availability.json`; исходный исполняемый зонд: `timing_probe.lua`.
Доступность относится к установленной Dota, не является обещанием для любого другого юнита или версии игры.

| Величина / место в коде | Что есть в ggd-v2 | Честный эквивалент Dota и область | Вычисление из публичного состояния симулятора |
| --- | --- | --- | --- |
| `UNIT_X/Y`, расстояние, направление, facing | x/y/dx/dy, facing_cos/sin | GetLocation, GetFacing; Св/Вр/К/Б | Да |
| `UNIT_HEALTH/MAX`, deny <= 50%, alive/ACTIVE | health/max/fraction, valid, attack_mask | GetHealth/GetMaxHealth, IsAlive; Св/Вр/К/Б | Да |
| TEAM/KIND, поколение цели | team_relative, type flags; generation не числовой вход | GetTeam/GetUnitName, стабильный handle до удаления; Св/Вр/К/Б | Да, жизненный цикл handle; ID не предсказывает будущее |
| `UNIT_ATTACK_DAMAGE`, min/max и разброс | attack_damage + combat.damage_spread | GetAttackDamage; Св/Вр/К/Б. Разброс — калиброванные статические правила | Да; точный будущий roll неизвестен обеим сторонам |
| `UNIT_ARMOR`, aura/множитель типа, округление урона | armor, type, incoming damage | GetArmor; Св/Вр/К/Б. Типовые множители/ceil — калибровка | Да при тех же статических правилах |
| `UNIT_HEALTH_REGEN`, `FLAG_AURA_END` | health_regen; суммарный текущий реген | GetHealthRegen; Св/Вр/К/Б | Да; нельзя угадывать будущий выход из ауры |
| `UNIT_ATTACK_RANGE`, hull для `attack_reach` | attack_range; hull не явный вход | GetAttackRange; Св/Вр/К/Б. GetBoundingRadius подтверждён для Св/Вр/К/Б; трактовка reach — статическая калибровка | Да при калиброванном hull |
| `UNIT_MOVE_SPEED`, approach | move_speed и позиции | GetCurrentMovementSpeed; Св/Вр/К/Б | Да, гипотеза о сохранении движения |
| `hero[6]`, поворот к цели | facing + геометрия; turn_rate не явный вход | GetTurnRate недоступен: 0 успешных вызовов на Св/Вр/К/Б (`api-availability-holdout.json`). GetFacing доступен; rate SF — статическая калибровка | Да для собственной SF по калибровке |
| `UNIT_ATTACK_POINT`, интервал и базовый интервал | attack_point, attack_interval | GetAttackPoint/GetSecondsPerAttack; Св/Вр/К/Б, зонд | Да; point_scaled = point * interval / base |
| `ATTACK_READY` | Нет явного таймера | GetLastAttackTime + interval - scaled point; Св/Вр/К/Б, зонд. LastAttackTime — момент release в шкале GameTime | Приближение из публичной истории release; изменение скорости атаки требует осторожности |
| `WINDUP`, оставшийся замах | is_attacking; attack_cycle в симуляторе всегда 0 | GetAnimActivity/GetAnimCycle; Св/Вр/К/Б, зонд и кривые. Это оценка с точностью около тика, не чтение скрытого таймера | Да как калиброванная оценка; ggd-v3 моделирует анимацию из внутреннего времени |
| `TARGET`, `WINDUP_TARGET`, поколения | combat.attack_target_dx/dy/health; attack_target_self/enemy | GetAttackTarget; Св/Вр/К/Б, зонд. Не гарантирует сохранение цели в будущем | Да, видимая цель; точный target замаха после отмены может быть недоступен |
| SHOTS: цель, caster, позиция, скорость, тип | projectiles + projectile_combat + агрегаты incoming | GetIncomingTrackingProjectiles на принимающем юните; Св/Вр/К/Б, зонд | Да для наблюдаемых снарядов; скрытый rolled damage не используется |
| `seen_attacks`: последний наблюдаемый урон/тип caster | combat уже хранит публичный cache; память LSTM | История GetAttackDamage/GetUnitName; Св/Вр/К/Б | Да, только прошлое наблюдение; prototype fallback — предположение |
| Flight, muzzle offsets, радиус попадания 8, 30 Hz ceil | позиции, скорость; формула не явный вход | Публичная геометрия + измеренные muzzle/скорость/тик; Св/Вр/К/Б | Да, приближение; стрелять по будущей позиции нельзя |
| Задержка исполнения 7/30 с, horizon 96 тиков | previous; задержка не отдельное поле | Известная конфигурация собственного executor | Да, фиксированные публичные параметры |
| `META_ATTACK_POINT_ERROR`, дробный остаток округления SF | Нет | Прямого Bot API нет; можно оценивать по истории начал/release, точность не доказана | Скрытый точный остаток — только oracle |
| `source_death`, suppression будущих атак умирающего источника | HP, текущие снаряды; результат не явный вход | Вычисляемый прогноз из тех же публичных наблюдений | Да приближённо; зависит от фаз/целей и min damage |
| damage[t], uncertainty[t], HP к hit, deadline смерти, sigma | incoming_damage/DPS и next_projectile_eta; нет полного временного ряда | Нет готового API. Возможен прогноз по публичным фазам/целям/урону | С честными фазами приближённо; точный teacher forecast включён только в oracle |
| own_shot, исключение повторного приказа | Видимые projectile caster/target и previous | GetIncomingTrackingProjectiles + собственная история | Да |
| Командная память `mem[1..6]`: цель, generation, queue release, hit, predicted HP | previous последней команды + LSTM | История собственных выданных команд и известная задержка | Да; численный teacher cache не дан явно честному входу |
| Выбор score = deadline + delay*.25 + sigma*.02 + deny penalty6, stable ID tie | Входы частично; готового score нет | Алгоритм учителя, не Bot API | Вычисляется при наличии прогнозов; oracle не содержит score, viable или выбранный label |
| protected/отмена doomed windup, пауза предмета `mem[14]` | previous, active statuses; нет явного lock | История своих приказов; публичные HP/замах/цель | Да приближённо; oracle forecast учитывает предыдущую команду, не обязательно «новая атака сейчас» |
| retreat hysteresis `mem[15]`, hp<.30 / >.56, threats | health_fraction, attack_target_self, combat | GetHealth/MaxHealth, GetAttackTarget; Св/Вр/К/Б | Да через историю; угрозы подавляют ATTACK |
| tango/flask/stash/courier counts, cooldown, item/buy masks, fear/action_mask | items, status, legal masks | Собственный inventory/модификаторы и законность действий | Да; покупки/лечение могут заменить ATTACK |
| centroid волны, tower tanks/дистанция, fallback MOVE/STOP, текущий ORDER | позиции/HP/команда previous; текущий ORDER не полный вход | Публичные позиции/HP и история собственных приказов | Да; влияет на достижимость будущего LH |
| episode/reset/audit terminal | reset, own health/valid | Собственная смерть/начало матча/штатный конец | Да; gate выполнения, не дополнительная информация о LH |

**Подтверждено кодом:** ggd-v2 уже содержит конкретную цель атаки через combat-геометрию, а не только две hero-target маски.
**Гипотеза:** основной информационный разрыв — фаза замаха и точная готовность; необходимость готового HP-прогноза проверяет эксперимент, а не эта таблица.
