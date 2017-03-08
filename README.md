# ZEN

This is an implementation of Drake Deming's Pixel-Level Decorrelation (PLD).

To clone the repo:

  git clone --recursive https://github.com/rychallener/ZEN zen

You have to compile the MCcubed package (https://github.com/pcubillos/MCcubed):

  cd zen
  cd mccubed
  make
  cd ..

To run the code, first copy all p3, p4, and p5 data outputs from POET to the zen directory. Then edit zen.cfg to your liking and execute with

  zen.py <eventname> zen.cfg

where <eventname> is the event code (e.g. wa029bs11).