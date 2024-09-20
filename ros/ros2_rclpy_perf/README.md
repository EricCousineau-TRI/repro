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
Sub running
Pub running, target rate: 2000.0
Pub + sub ready. Stopping after 1.0

Pub done after 1.00037 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
pub.message_0         1999.052      0.0000508      1999      0.0002765      0.0012063

Sub done after 1.00042 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
sub.message_0         1999.489      0.0000741      1998      0.0002359      0.0012683


+ ./pub_sub.py --count 5
Sub running
Pub running, target rate: 2000.0
Pub + sub ready. Stopping after 1.0

Pub done after 1.00055 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
pub.message_0         2000.836      0.0000422      2001      0.0002215      0.0009059
pub.message_1         2000.691      0.0000426      2000      0.0002132      0.0008962
pub.message_2         2000.710      0.0000428      2000      0.0002155      0.0008742
pub.message_3         2000.749      0.0000434      2000      0.0002145      0.0009046
pub.message_4         2000.757      0.0000460      2000      0.0002003      0.0011380

Sub done after 1.00213 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
sub.message_0         1939.490      0.0000838      1938      0.0002367      0.0010471
sub.message_1         1858.463      0.0000942      1857      0.0003119      0.0010470
sub.message_2         1746.102      0.0001163      1744      0.0003859      0.0012390
sub.message_3         1652.004      0.0001390      1650      0.0003862      0.0012429
sub.message_4         1543.878      0.0001704      1542      0.0003890      0.0012972


+ ./pub_sub.py --count 10
Sub running
Pub running, target rate: 2000.0
Pub + sub ready. Stopping after 1.0

Pub done after 1.00058 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
pub.message_0         2000.099      0.0000387      1999      0.0003730      0.0006384
pub.message_1         2000.122      0.0000392      1999      0.0003728      0.0006441
pub.message_2         2000.133      0.0000396      1999      0.0003717      0.0006429
pub.message_3         2000.150      0.0000396      1999      0.0003704      0.0006399
pub.message_4         2000.161      0.0000396      1999      0.0003696      0.0006361
pub.message_5         2000.173      0.0000398      1999      0.0003693      0.0006391
pub.message_6         2000.185      0.0000400      1999      0.0003693      0.0006622
pub.message_7         2000.201      0.0000402      1999      0.0003692      0.0006903
pub.message_8         2000.228      0.0000405      1999      0.0003687      0.0007181
pub.message_9         2000.243      0.0000407      1999      0.0003688      0.0007443

Sub done after 1.00094 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
sub.message_0         1133.471      0.0001045      1132      0.0004337      0.0010702
sub.message_1         1133.493      0.0001044      1132      0.0004303      0.0010700
sub.message_2         1131.987      0.0001042      1130      0.0005986      0.0013915
sub.message_3         1130.987      0.0001053      1129      0.0005993      0.0014420
sub.message_4         1067.877      0.0001688      1066      0.0006709      0.0015443
sub.message_5          921.628      0.0002619       920      0.0006735      0.0016493
sub.message_6          884.564      0.0002766       883      0.0007467      0.0017165
sub.message_7          866.532      0.0002870       865      0.0008342      0.0017883
sub.message_8          855.516      0.0002954       854      0.0008201      0.0019571
sub.message_9          794.409      0.0003366       793      0.0008203      0.0019637


+ ./pub_sub.py --count 15
Pub running, target rate: 2000.0
Sub running
Pub + sub ready. Stopping after 1.0

Pub done after 1.00209 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
pub.message_0         1841.513      0.0000375      1841      0.0005291      0.0008813
pub.message_1         1841.542      0.0000375      1841      0.0005293      0.0008811
pub.message_2         1841.556      0.0000374      1841      0.0005295      0.0008815
pub.message_3         1841.563      0.0000374      1841      0.0005296      0.0008804
pub.message_4         1841.572      0.0000374      1841      0.0005297      0.0008819
pub.message_5         1841.598      0.0000375      1841      0.0005299      0.0008824
pub.message_6         1841.607      0.0000375      1841      0.0005298      0.0008836
pub.message_7         1841.593      0.0000376      1840      0.0005297      0.0008803
pub.message_8         1841.605      0.0000377      1840      0.0005292      0.0008790
pub.message_9         1841.611      0.0000377      1840      0.0005295      0.0008798
pub.message_10        1841.616      0.0000377      1840      0.0005293      0.0008796
pub.message_11        1841.625      0.0000376      1840      0.0005293      0.0008802
pub.message_12        1841.638      0.0000376      1840      0.0005295      0.0008805
pub.message_13        1841.645      0.0000375      1840      0.0005292      0.0008803
pub.message_14        1841.655      0.0000374      1840      0.0005294      0.0008808

Sub done after 1.00043 sec
                Mean Freq (hz)     Stddev (s)   Samples        Min (s)        Max (s)
sub.message_0          685.285      0.0001056       685      0.0004967      0.0017491
sub.message_1          684.349      0.0001005       683      0.0011257      0.0019180
sub.message_2          684.626      0.0000991       683      0.0011215      0.0017543
sub.message_3          684.624      0.0000991       683      0.0011241      0.0017563
sub.message_4          684.624      0.0000990       683      0.0011244      0.0017592
sub.message_5          684.622      0.0000991       683      0.0011221      0.0017604
sub.message_6          684.618      0.0000995       683      0.0011248      0.0017599
sub.message_7          683.607      0.0001054       682      0.0011282      0.0024173
sub.message_8          681.611      0.0001186       680      0.0011233      0.0025166
sub.message_9          680.610      0.0001268       679      0.0011235      0.0026696
sub.message_10         679.605      0.0001343       678      0.0011779      0.0026689
sub.message_11         598.410      0.0004518       597      0.0011787      0.0028882
sub.message_12         506.193      0.0006037       505      0.0012642      0.0029888
sub.message_13         495.163      0.0006114       494      0.0013369      0.0029958
sub.message_14         490.155      0.0006143       489      0.0013862      0.0030114
```
