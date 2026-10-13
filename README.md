# BIKV: структурно-осознанное сжатие KV-cache

Исследовательский код курсовой работы «Исследование обработки длинного контекста в больших языковых моделях».

Проект проверяет, достаточно ли оценивать фрагменты длинного документа независимо или при ограниченном KV-cache необходимо учитывать совместную полезность нескольких блоков. Предлагаемый метод BIKV (Boundary-Interaction KV compression) объединяет query-aware unary-признаки с признаками произвольных пар блоков и выбирает маску при фиксированном бюджете памяти.

## Исследовательский вопрос

Пусть документ разбит на `N` блоков, а в KV-cache можно сохранить только `B < N` блоков. Обычный селектор предполагает аддитивную полезность:

```text
F_unary(S) = sum(i in S) u_i
```

BIKV добавляет совместный член:

```text
F_pair(S) = sum(i in S) u_i + sum((i, j) in E(S)) p_ij
```

Основная проверяемая гипотеза: pair-признаки помогают сохранять распределённые evidence-цепочки, которые могут быть потеряны при независимом ранжировании блоков.

## Что реализовано

- разбиение документа на блоки фиксированной длины и exact-budget выбор маски;
- причинно чистый протокол `document-compress-then-query`;
- query-attention, KNorm, block-level и arbitrary-pair признаки;
- факторная разметка взаимодействия через маски `M11`, `M10`, `M01`, `M00`;
- matched unary Ridge и unary+pair Ridge с одинаковыми candidate masks, target, folds и регуляризацией;
- shuffled-pair control, разрушающий только соответствие между pair-признаком и реальной связью;
- token- и block-level KNorm, query-attention, random, task oracle и matched-budget SnapKV;
- диагностический `relation_graph` для оценки структурного headroom;
- family bootstrap, paired bootstrap и wins/losses/ties;
- атомарные checkpoints и возобновление Colab/Kaggle-запусков.

Диагностический query-pass используется только для построения признаков и полностью удаляется. Затем выбранные блоки заново проходят prefill, и вопрос впервые обрабатывается против сжатого document cache. Это исключает перенос удалённого факта через hidden states вопроса.

`relation_graph` не является learned BIKV и не использует gold answer, но эксплуатирует явную лексическую связность текста. Его результаты приводятся отдельно как оценка потенциала graph-aware отбора.

## Экспериментальная программа

Внутренние номера версий и этапов сохранены только в именах frozen-файлов для provenance. Научно эксперименты организованы следующим образом.

| Эксперимент | Данные | Контекст и бюджет | Основной вопрос |
|---|---|---|---|
| Проверка backbone | 48 counterfactual families, 144 варианта | 14 блоков | Решает ли модель оба hop и зависит ли ответ от evidence? |
| Факторная разметка | Контролируемые двухшаговые истории | Exact-budget `M11/M10/M01/M00` | Существует ли измеримый неаддитивный pair-effect? |
| Средний тест | 48 новых историй | 42 блока, `B=18` | Корректен ли полный pipeline и возникает ли насыщение? |
| Stress-тест бюджета | Те же замороженные captures | `B=6/10/18` | При каком ограничении методы становятся различимы? |
| Масштабирование длины | По 64 свежих истории | 2k/4k/8k, fixed-memory и fixed-ratio | Переносится ли frozen selector между длинами? |
| Произвольные цепочки | 36 историй, цепочки длины 2--4 | Около 4k, `B=6` | Сохраняет ли pair-aware отбор всю распределённую цепочку? |
| RULER Variable Tracking | 50 примеров на длину | 4k и 8k | Переносится ли метод на независимую распределённую задачу? |
| LongBench-E 2Wiki | Два непересекающихся cohort по 24 | 3--6k, `B=6` | Переносится ли frozen selector на естественный multi-document QA? |

## Итоговые результаты

### Контролируемые эксперименты

| Условие и метрика | BIKV pair | Matched unary | SnapKV | Full / oracle | Вывод |
|---|---:|---:|---:|---:|---|
| Средний тест 4k, `B=18`, full-correct accuracy | **100.0%** | **100.0%** | 93.5% | Full 100.0% | Потолочный эффект, pair и unary неразличимы |
| Budget stress 4k, `B=6`, accuracy | **87.0%** | 82.6% | 76.1% | - | Pair получает локальный выигрыш при жёстком бюджете |
| Length 2k, ratio `B=9`, full-correct accuracy | 59.6% | 55.8% | **80.8%** | - | Pair немного выше unary, но уступает SnapKV |
| Length 4k, ratio `B=18`, full-correct accuracy | 85.4% | 80.5% | **87.8%** | - | Лучший режим переноса pair относительно unary |
| Length 8k, ratio `B=36`, full-correct accuracy | 91.7% | **94.4%** | 77.8% | - | Универсального роста pair-effect с длиной нет |
| Цепочки 4k, complete-chain retention | **30.6%** | 0.0% | - | Graph diagnostic 100.0% | Pair лучше сохраняет полную цепочку |
| Цепочки 4k, candidate accuracy | 16.7% | 16.7% | 25.0% | Graph diagnostic 91.7% | Улучшение retention не перешло в answer accuracy |

Для контролируемых цепочек прирост complete-chain retention относительно matched unary составил `+30.6 п.п.` с 95% CI `[11.1; 52.8]`. Разность candidate accuracy при этом равна нулю: селектор чаще сохраняет всю цепочку, но Qwen2.5-3B не всегда использует оставленные факты.

### Внешние benchmark

| Набор и метрика | Full context | BIKV pair | Matched unary | SnapKV | Shuffled pair |
|---|---:|---:|---:|---:|---:|
| RULER 4k, Set F1 | **80.9%** | 25.3% | 27.8% | 74.4% | 25.3% |
| RULER 8k, Set F1 | 34.7% | 32.0% | 32.0% | **37.3%** | 32.0% |
| LongBench cohort A, answer F1, `N=24` | **55.4%** | 43.7% | 42.9% | 46.1% | 13.5% |
| LongBench cohort B, answer F1, `N=24` | 59.6% | 59.7% | 54.2% | 57.5% | 37.5% |
| LongBench вместе, answer F1, `N=48` | **57.5%** | 51.7% | 48.5% | 51.8% | 25.5% |

### Главные парные сравнения

| Сравнение | Разность | 95% CI | Wins / losses / ties | Интерпретация |
|---|---:|---:|---:|---|
| BIKV - matched unary, LongBench `N=48` | `+3.2 п.п.` | `[-10.7; 17.0]` | `9 / 6 / 33` | Направление положительное, преимущество не доказано |
| BIKV - shuffled pair, LongBench `N=48` | **`+26.2 п.п.`** | **`[13.5; 39.2]`** | `18 / 1 / 29` | Настоящее соответствие pair-признаков полезно |
| BIKV - SnapKV, LongBench `N=48` | `-0.1 п.п.` | `[-13.7; 13.4]` | `8 / 8 / 32` | Методы практически совпали по F1 |
| BIKV - matched unary, chain retention | **`+30.6 п.п.`** | **`[11.1; 52.8]`** | - | Pair-aware отбор лучше сохраняет evidence-цепочку |

Итоговый вывод ограничен: pair-сигнал воспроизводим и улучшает сохранение структуры, но BIKV не показал универсального превосходства над matched unary и SnapKV. На RULER текущая pair-модель переносится плохо; на LongBench она практически совпадает со SnapKV и устойчиво превосходит только shuffled control.

## Структура репозитория

```text
src/bikv/   упаковка контекста, cache-протокол, признаки, селекторы, метрики и storage
scripts/    подготовка данных, GPU-runner, offline-обучение и анализ результатов
configs/    frozen-конфигурации, веса селекторов и manifests независимых выборок
data/       замороженные confirmation-наборы length sweep, цепочки и их аудиты
notebooks/  воспроизводимые Colab/Kaggle-запуски основных экспериментов
docs/       подробные протоколы независимых проверок
tests/      CPU-тесты cache-инвариантов, масок, признаков, метрик и resume-логики
```

Скачанные модели, временные файлы, сырые captures и архивы запусков в репозиторий не входят. Frozen-веса линейных селекторов, confirmation datasets и manifests выборок включены, поскольку необходимы для воспроизведения независимых тестов. Повторный budget stress использует F2 captures из исходного GPU-артефакта; без этого архива notebook воспроизводит код анализа, но не пересчитывает исторические маски.

## Локальная проверка

Требуется Python 3.10 или новее.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
```

CPU достаточно для unit-тестов и offline-анализа готовых captures. Полные KV-интервенции рассчитаны на NVIDIA T4 с 15--16 GB VRAM либо более мощную GPU.

## Запуски Colab и Kaggle

Основные notebooks:

- [Проверка backbone и факторная разметка](notebooks/BIKV_Colab_Kaggle.ipynb)
- [Средний тест на 42 блоках](notebooks/BIKV_V12_Stage_F2_Medium_Colab.ipynb)
- [Stress-тест бюджета](notebooks/BIKV_V13_Stage_G1_Budget_Colab.ipynb)
- [Length sweep: 2k](notebooks/BIKV_V16_G1b_2k_Confirmation_Colab_Kaggle.ipynb)
- [Length sweep: 4k](notebooks/BIKV_V16_G1b_4k_Confirmation_Colab_Kaggle.ipynb)
- [Length sweep: 8k](notebooks/BIKV_V16_G1b_8k_Confirmation_Colab_Kaggle.ipynb)
- [RULER Variable Tracking: 4k](notebooks/BIKV_V16_G2_RULER_4k_Comparison_Colab_Kaggle.ipynb)
- [RULER Variable Tracking: 8k](notebooks/BIKV_V16_G2_RULER_8k_Comparison_Colab_Kaggle.ipynb)
- [Контролируемые произвольные цепочки](notebooks/BIKV_V17_Stage_H_Chains_4k_Colab_Kaggle.ipynb)
- [LongBench-E 2Wiki, первый cohort](notebooks/BIKV_V18_Stage_I_LongBench_2Wiki_Colab_Kaggle.ipynb)
- [LongBench-E 2Wiki, подтверждающий cohort](notebooks/BIKV_V19_Stage_I_B_LongBench_2Wiki_Replication_Colab_Kaggle.ipynb)

Версионные номера в именах notebooks фиксируют provenance выполненных запусков и не являются структурой научного описания.

Для length sweep, RULER, chain и LongBench notebooks можно собрать загрузочные архивы:

```bash
python scripts/build_release.py
python scripts/build_stage_h_release.py
python scripts/build_stage_i_release.py
python scripts/build_stage_i_b_release.py
```

`build_release.py` создаёт continuation-архив с frozen length datasets и конфигурациями RULER. Остальные builders создают независимые архивы для chain и двух LongBench cohorts.

В Colab результаты зеркалируются в `MyDrive/bikv_runs`; в Kaggle они сохраняются в `/kaggle/working`. Повторный запуск восстанавливает завершённые строки и captures и продолжает расчёт без их повторения.

## Научная интерпретация

Главное сравнение learned-селекторов:

```text
graph_arbitrary_pair_ridge_v1
vs
graph_matched_unary_ridge_v1
```

Обе модели используют один candidate pool, target, folds, Ridge-регрессию и регуляризацию. Поэтому различие изолирует вклад pair-признаков. Сравнение с `graph_pair_shuffled_v1` проверяет, важна ли настоящая структура, а не просто дополнительные числовые признаки.

Проект не утверждает, что BIKV является готовой заменой SnapKV. Его научный результат состоит в разделении индивидуальной, парной и графовой важности при сжатии длинного контекста и в установлении условий, при которых pair-aware представление помогает или перестаёт переноситься.

## Состояние проекта

Реализация и запланированные GPU-эксперименты завершены. Репозиторий содержит код основных и подтверждающих запусков, frozen-конфигурации, manifests выборок и тесты. Сырые результаты запусков хранятся отдельно из-за размера.
