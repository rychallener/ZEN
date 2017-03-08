# ZEN

This is an implementation of Drake Deming's Pixel-Level Decorrelation (PLD).

To clone the repo:
```shell
  git clone --recursive https://github.com/rychallener/ZEN zen
```

You have to compile the MCcubed package (https://github.com/pcubillos/MCcubed):
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