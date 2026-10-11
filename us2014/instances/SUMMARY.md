## Summary table

| instance | flights | tails | airports | itineraries | pax | delay disr. | cancel disr. | aircraft outages | airport reductions | maintenance | fixed flights | repair drop rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| US02 | 9,755 | 1034 | 238 | 68,267 | 552,074 | 935 | 239 | 21 | 123 | 59 | 1255 | 8.9% |
| US03 | 777 | 40 | 34 | 1,142 | 13,952 | 48 | 18 | 2 | 18 | 3 | 107 | 7.4% |
| US04 | 9,214 | 1034 | 240 | 64,031 | 537,892 | 849 | 262 | 24 | 78 | 40 | 1178 | 7.9% |
| US05 | 9,244 | 1027 | 238 | 66,910 | 542,469 | 1200 | 413 | 28 | 202 | 36 | 1012 | 11.7% |
| US06 | 10,956 | 1030 | 231 | 81,727 | 674,276 | 713 | 60 | 12 | 33 | 74 | 1322 | 5.5% |
| US07 | 11,407 | 1048 | 243 | 88,178 | 731,805 | 1323 | 291 | 25 | 119 | 57 | 1552 | 7.9% |
| US08 | 11,111 | 1048 | 248 | 86,019 | 703,384 | 1213 | 96 | 17 | 54 | 64 | 1571 | 7.0% |
| US09 | 10,672 | 1036 | 235 | 83,632 | 648,497 | 892 | 117 | 6 | 35 | 82 | 1166 | 6.6% |
| US10 | 10,766 | 1043 | 233 | 84,328 | 663,858 | 244 | 3 | 7 | 2 | 108 | 1360 | 4.5% |
| US02_CRJ700 | 684 | 67 | 59 | 1,868 | 34,296 | 72 | 4 | 5 | 44 | 3 | 88 | 9.3% |
| US02_A320 | 935 | 140 | 48 | 3,735 | 96,504 | 76 | 0 | 0 | 73 | 12 | 123 | 8.4% |
| US02_B737 | 1,529 | 232 | 59 | 8,122 | 179,149 | 134 | 1 | 1 | 70 | 9 | 204 | 4.7% |
| US02_CRJ200 | 2,160 | 150 | 115 | 4,555 | 78,217 | 185 | 23 | 6 | 44 | 7 | 282 | 5.8% |
| US08_EMB120 | 732 | 41 | 36 | 1,012 | 14,330 | 59 | 8 | 4 | 10 | 4 | 96 | 7.6% |
| US08_CRJ700 | 749 | 67 | 68 | 2,064 | 41,277 | 103 | 4 | 1 | 36 | 4 | 97 | 11.7% |
| US08_A320 | 1,270 | 146 | 58 | 5,927 | 139,936 | 164 | 0 | 0 | 34 | 8 | 199 | 7.0% |

## Validation

| instance | loader round-trip | twin n_violations | twin pos_missing | twin seat spill | bad connections | itineraries referencing dropped flights | repair rerun drops | capacity-fix rerun updates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| US02 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US03 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US04 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US05 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US06 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US07 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US08 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US09 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US10 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US02_CRJ700 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US02_A320 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US02_B737 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US02_CRJ200 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US08_EMB120 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US08_CRJ700 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| US08_A320 | OK | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## FH gate (scale-down instances, 10 min limit, HiGHS, LP-restricted)

| instance | flights | status | obj | bound | gap | wall s | solver n_violations | gate |
|---|---:|---|---:|---:|---:|---:|---:|---|
| US03 | 777 | Optimal | 1,177,136 | 1,167,711 | 8.0e-03 | 9.0 | 0 | solver-feasible |
| US02_CRJ700 | 684 | Optimal | 2,511,924 | 2,511,924 | 0.0e+00 | 7.6 | 0 | solver-feasible |
| US02_A320 | 935 | Optimal | 8,261,830 | 8,243,812 | 2.2e-03 | 34.7 | 0 | solver-feasible |
| US02_B737 | 1529 | Time limit reached | 43,496,440 | 14,473,126 | 6.7e-01 | 618.4 | 1 | hard (witness-feasible) |
| US02_CRJ200 | 2160 | Optimal | 5,339,482 | 5,339,443 | 7.4e-06 | 61.4 | 0 | solver-feasible |
| US08_EMB120 | 732 | Optimal | 1,072,815 | 1,072,815 | -6.5e-16 | 7.4 | 0 | solver-feasible |
| US08_CRJ700 | 749 | Optimal | 2,432,000 | 2,432,000 | 1.3e-15 | 3.6 | 0 | solver-feasible |
| US08_A320 | 1270 | Optimal | 6,180,221 | 6,178,902 | 2.1e-04 | 24.5 | 0 | solver-feasible |

## Full-year flight count vs the wiki

| filter on 2014 OTP rows | flights | vs wiki 1,164,155 |
|---|---:|---:|
| reporting carrier UA+EV, any tail | 1,179,546 | +15,391 (+1.3%) |
| reporting carrier UA+OO, any tail | 1,106,558 | -57,597 (-4.9%) |
| reporting carrier EV+OO, any tail | 1,299,048 | +134,893 (+11.6%) |
| reporting carrier UA+OO, tail in United fleet file | 993,726 | -170,429 (-14.6%) |
| **our base filter**: carriers UA+EV+OO and tail in `combinedPlaneData` | **1,441,228** | +277,073 (+23.8%) |

OTP 2014 rows for UA/EV/OO: 1,792,579 (UA 493,528, EV 686,021, OO 613,030); rows without tail number: 7,343.
