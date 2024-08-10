# rclpy pub / sub stats

Results on laptop, Ubuntu 22.04:

```sh
$ ./repro.sh
+ grep 'model name' /proc/cpuinfo
+ uniq
model name      : Intel(R) Xeon(R) W-10855M CPU @ 2.80GHz
+ nproc
12
+ python3 -c 'import rclpy; print(rclpy.utilities.get_rmw_implementation_identifier())'
rmw_cyclonedds_cpp
+ ./pub_sub.py --count 1
Pub running, target rate: 2000.0
Sub running
Pub + sub ready. Stopping after 1.0

Pub done after 1.00185 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
pub.message_0         1997.311      0.0000630      1997      0.0000437      0.0021618

Sub done after 1.00039 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
sub.message_0         1996.001      0.0000769      1994      0.0003141      0.0022862


+ ./pub_sub.py --count 5
Pub running, target rate: 2000.0
Sub running
Pub + sub ready. Stopping after 1.0

Pub done after 1.00822 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
pub.message_0         2000.200      0.0000299      2000      0.0003304      0.0006065
pub.message_1         2000.255      0.0000303      2000      0.0003213      0.0006124
pub.message_2         2000.322      0.0000305      1999      0.0002994      0.0006158
pub.message_3         2000.380      0.0000309      1999      0.0002803      0.0006387
pub.message_4         2000.429      0.0000313      1999      0.0002696      0.0006648

Sub done after 1.00056 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
sub.message_0         1837.659      0.0001114      1837      0.0003326      0.0010176
sub.message_1         1611.162      0.0001430      1609      0.0004093      0.0010225
sub.message_2         1458.945      0.0001522      1457      0.0004111      0.0010698
sub.message_3         1367.836      0.0001646      1366      0.0004698      0.0017055
sub.message_4         1321.792      0.0001752      1320      0.0004941      0.0017159


+ ./pub_sub.py --count 10
Sub running
Pub running, target rate: 2000.0
Pub + sub ready. Stopping after 1.0

Pub done after 1.00028 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
pub.message_0         2000.303      0.0000428      2000      0.0003673      0.0006023
pub.message_1         2000.323      0.0000431      2000      0.0003672      0.0006175
pub.message_2         2000.346      0.0000435      2000      0.0003669      0.0006217
pub.message_3         2000.351      0.0000441      2000      0.0003666      0.0006316
pub.message_4         2000.361      0.0000442      2000      0.0003666      0.0006409
pub.message_5         2000.540      0.0000440      1999      0.0003668      0.0006520
pub.message_6         2000.550      0.0000442      1999      0.0003674      0.0006632
pub.message_7         2000.560      0.0000444      1999      0.0003677      0.0006740
pub.message_8         2000.563      0.0000443      1999      0.0003673      0.0006845
pub.message_9         2000.584      0.0000447      1999      0.0003671      0.0006933

Sub done after 1.00096 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
sub.message_0         1000.856      0.0000608      1000      0.0006252      0.0015486
sub.message_1         1000.878      0.0000602      1000      0.0006179      0.0014867
sub.message_2         1000.884      0.0000600      1000      0.0006186      0.0014218
sub.message_3          999.890      0.0000621       999      0.0007152      0.0016176
sub.message_4          998.426      0.0000655       997      0.0007137      0.0017358
sub.message_5          993.422      0.0000822       992      0.0007122      0.0019095
sub.message_6          981.422      0.0001160       979      0.0007890      0.0018459
sub.message_7          954.362      0.0001778       952      0.0008599      0.0019025
sub.message_8          908.250      0.0002574       906      0.0009309      0.0019575
sub.message_9          798.982      0.0003926       797      0.0009467      0.0026422


+ ./pub_sub.py --count 15
Pub running, target rate: 2000.0
Sub running
Pub + sub ready. Stopping after 1.0

Pub done after 1.00117 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
pub.message_0         1734.620      0.0000653      1734      0.0005354      0.0010239
pub.message_1         1734.622      0.0000653      1734      0.0005355      0.0010288
pub.message_2         1734.614      0.0000653      1734      0.0005358      0.0010379
pub.message_3         1734.619      0.0000652      1734      0.0005355      0.0010452
pub.message_4         1734.610      0.0000652      1734      0.0005356      0.0010479
pub.message_5         1734.600      0.0000652      1734      0.0005357      0.0010441
pub.message_6         1734.601      0.0000653      1734      0.0005358      0.0010381
pub.message_7         1734.793      0.0000652      1733      0.0005356      0.0010339
pub.message_8         1734.783      0.0000652      1733      0.0005355      0.0010295
pub.message_9         1734.773      0.0000652      1733      0.0005355      0.0010241
pub.message_10        1734.767      0.0000651      1733      0.0005353      0.0010190
pub.message_11        1734.758      0.0000651      1733      0.0005355      0.0010208
pub.message_12        1734.747      0.0000651      1733      0.0005354      0.0010214
pub.message_13        1734.740      0.0000652      1733      0.0005353      0.0010215
pub.message_14        1734.730      0.0000652      1733      0.0005354      0.0010218

Sub done after 1.00043 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
sub.message_0          722.808      0.0001238       722      0.0007867      0.0021237
sub.message_1          722.827      0.0001240       722      0.0007812      0.0021260
sub.message_2          722.835      0.0001241       722      0.0007785      0.0021278
sub.message_3          722.838      0.0001241       722      0.0007777      0.0021282
sub.message_4          721.840      0.0001260       721      0.0007785      0.0021881
sub.message_5          721.404      0.0001236       720      0.0009237      0.0021887
sub.message_6          721.405      0.0001231       720      0.0009246      0.0021885
sub.message_7          720.408      0.0001257       719      0.0010747      0.0023101
sub.message_8          717.405      0.0001426       716      0.0010843      0.0025452
sub.message_9          713.444      0.0001619       711      0.0010845      0.0024843
sub.message_10         699.397      0.0002282       697      0.0011496      0.0026304
sub.message_11         637.184      0.0004183       635      0.0012254      0.0030394
sub.message_12         519.782      0.0006073       518      0.0013033      0.0032121
sub.message_13         449.544      0.0006134       448      0.0013798      0.0032604
sub.message_14         422.453      0.0005795       421      0.0013843      0.0033096
```
