### ZEN

This is an implementation of Drake Deming's Pixel-Level Decorrelation (PLD).

### Team Members:
* [Ryan Challener](https://github.com/rychallener/) (UCF) <rchallen@knights.ucf.edu>
* Andrew Foster (UCF)
* Em DeLarme (UCF)

To clone the repo:
```shell
  git clone --recursive https://github.com/rychallener/ZEN zen
```

You have to compile the [MCcubed](https://github.com/pcubillos/MCcubed) package:
```shell
  cd zen
  cd mccubed
  make
  cd ..
```

To run the code, first copy all p3, p4, and p5 data outputs from POET to the zen directory. Then edit zen.cfg to your liking and execute with
```shell
  zen.py <eventname> zen.cfg
```

where <eventname> is the event code (e.g. wa029bs11).