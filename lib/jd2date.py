def jd2date(jd):
  """
  Convert a Julian date to Gregorian date

  Parameters:
  -----------
  jd: Float
     Julian date.

  Returns:
  --------
  date: String
     Gegorian date representation of jd (yyyy_mm_dd).

  Notes:
  ------
  Description at: http://mathforum.org/library/drmath/view/51907.html
  Original algorithm in Jean Meeus: "Astronomical Formulae for Calculators"

  Modification History:
  ---------------------
  2009-02-15  Ian Crossfield  Implemented in Python
  2014-04-14  patricio        Extracted from IC's routine.
  """

  jd = jd+0.5
  Z = int(jd)
  F = jd-Z
  alpha = int((Z-1867216.25)/36524.25)
  A = Z + 1 + alpha - int(alpha/4)
  
  B = A + 1524
  C = int( (B-122.1)/365.25)
  D = int( 365.25*C )
  E = int( (B-D)/30.6001 )
  
  dd = B - D - int(30.6001*E) + F
  dd = int(dd)
  
  if E<13.5:
    mm = E-1
  else:
    mm = E-13
  
  if mm>2.5:
    yyyy = C-4716
  else:
    yyyy = C-4715

  # Format the date into a string:
  date = "%04d_%02d_%02d"%(yyyy, mm, dd)

  return date
