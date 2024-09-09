For testing custom pickle backwards-compatibility

```sh
$ python ./old_code/main.py
$ python ./new_code/main.py
{'class_name': 'MyObject',
 'module_name': '__main__',
 'state': {'sub_dataclass': {'class_name': 'SubDataclass',
                             'module_name': '__main__',
                             'state': {'value': 10}},
           'sub_object': {'class_name': 'SubObject',
                          'module_name': '__main__',
                          'state': {'value': array([1., 2.])}},
           'sub_special': {'class_name': 'SubSpecialObject',
                           'module_name': '__main__',
                           'state': (30, 'special')},
           'value': 'abc'}}
```
