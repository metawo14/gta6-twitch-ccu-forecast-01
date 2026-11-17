"""gta6-twitch-ccu-forecast-01: hour-by-hour Twitch category viewers for GTA VI's release day.

The program reads one case file (makefun/twitch-ccu-case/v1) and writes, for hours 2..24 of the
release day (UTC), five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the category's concurrent
viewers. The score is the mean pinball loss in log space, so the model works in logs throughout.

Model (a curve-transfer ensemble):
  - hour 1's print p1 anchors every forecast;
  - f2 = ln(p1 / pre_peak) measures how far hour 1 already sits above the quiet pre-release
    evening peak (the busiest 19..23 UTC hour in the 7 days before the day), i.e. how much of the
    launch's rise the first hour already captured;
  - each comparable launch gives a training row: its own hour-1 print, its own f2, and the log
    ratio y_h = ln(viewers_h / p1_c) at each hour h;
  - per hour, three OLS ridge-free linear members on those rows (A "cap", B "quad", C "winsor"),
    each turned into quantiles through the quantiles of its residuals;
  - the three members are combined by a geometric mean in log space;
  - member C's predicted median is smoothed with a window-3 mean across hours (a launch curve is
    smooth, so the hourly fits should not jitter), the whole row shifted with it;
  - the result is scaled back by p1, forced non-decreasing across levels, clamped and rounded.

Standard library only, no randomness, no clock, no network: a pure function of the case JSON.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))
CAP = 250000.0
WINSOR_LO = 0.05
WINSOR_HI = 0.95


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def round2(x):
    """Round half up to 2 decimals, as JavaScript's Math.round(x * 100) / 100 does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile(sorted_values, p):
    """Linear-interpolation quantile (numpy's default, type 7)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = (n - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting on a small dense system."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return [0.0] * n
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for r in range(col + 1, n):
            factor = a[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        total = a[r][n]
        for c in range(r + 1, n):
            total -= a[r][c] * out[c]
        out[r] = total / a[r][r]
    return out


def ols(rows, target):
    """OLS with an intercept, normal equations solved by Gaussian elimination."""
    k = len(rows[0]) + 1
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for feats, y in zip(rows, target):
        design = [1.0] + list(feats)
        for i in range(k):
            xty[i] += design[i] * y
            for j in range(k):
                xtx[i][j] += design[i] * design[j]
    return solve(xtx, xty)


def predict(coef, feats):
    total = coef[0]
    for i, f in enumerate(feats):
        total += coef[i + 1] * f
    return total


def pre_peak_viewers(points, day_start):
    """Busiest 19..23 UTC hour in the 7 days before the day, or None when there is none."""
    lo = day_start - timedelta(days=7)
    best = None
    for at, viewers in points:
        if lo <= at < day_start and 19 <= at.hour <= 23:
            if best is None or viewers > best:
                best = viewers
    return best


def features_for(member, log_p1, f2):
    if member == "A":
        return [min(log_p1, math.log(CAP)), f2]
    if member == "B":
        return [log_p1, log_p1 * log_p1 / 100.0, f2]
    return [log_p1, f2]


def feature_ranges(rows):
    """Per-member [min, max] of every fitted feature over the training launches."""
    ranges = {}
    for member in ("A", "B", "C"):
        vectors = [features_for(member, log_p1, f2) for log_p1, f2, _ in rows]
        ranges[member] = [
            (min(v[i] for v in vectors), max(v[i] for v in vectors))
            for i in range(len(vectors[0]))
        ]
    return ranges


def clamp_features(feats, ranges):
    """Keep a target's features inside the range the fit saw, so no member extrapolates.

    The final case can sit far outside it: its category held a few thousand viewers before the
    launch, so f2 (how far hour 1 already rose above the quiet pre-release evening) is several
    log units above anything in the training launches, and an unclamped slope there multiplies
    hour 1 by a factor no category on Twitch reaches. Clamping costs nothing in range.
    """
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(feats, ranges)]


def training_rows(case):
    """One row per usable comparable: (log p1, f2, {hour: log ratio})."""
    day = parse_ts(case["day_start"])
    rows = []
    for comp in case["comparables"]:
        points = [(parse_ts(p["from"]), p["viewers"]) for p in comp["hourly"]]
        by_ts = dict(points)
        comp_day = parse_ts(comp["day_start"])
        comp_p1 = by_ts.get(comp_day)
        if comp_p1 is None or comp_p1 < 1:
            continue
        ratios = {}
        usable = True
        for h in HOURS:
            viewers = by_ts.get(comp_day + timedelta(hours=h - 1))
            if viewers is None:
                usable = False
                break
            ratios[h] = math.log(max(1, viewers) / max(1, comp_p1))
        if not usable:
            continue
        peak = pre_peak_viewers(points, comp_day)
        if peak is None:
            peak = comp_p1
        log_p1 = math.log(max(1, comp_p1))
        f2 = log_p1 - math.log(max(1, peak))
        rows.append((log_p1, f2, ratios))
    return rows


def fit_hour(rows, hour, member):
    """Fit one member for one hour: coefficients plus residual quantiles."""
    xs = []
    ys = []
    for log_p1, f2, ratios in rows:
        xs.append(features_for(member, log_p1, f2))
        ys.append(ratios[hour])
    if member == "C":
        ordered = sorted(ys)
        lo = quantile(ordered, WINSOR_LO)
        hi = quantile(ordered, WINSOR_HI)
        ys = [min(max(y, lo), hi) for y in ys]
    coef = ols(xs, ys)
    residuals = sorted(y - predict(coef, x) for x, y in zip(xs, ys))
    return {
        "coef": coef,
        "rq": [quantile(residuals, tau) for tau in LEVELS],
    }


# Baked fallback fits, derived in this build session by running this same three-member fit
# (training_rows + fit_hour) once over the kit p00 comparables (37 usable launches) and copying
# the per-hour coefficients and residual quantiles here. Used only when a case carries fewer
# than 10 usable comparables; the runtime fit is the normal path.
FALLBACK_FITS = {
     2: {"A": {"coef": [-0.1513650982, 0.0115791233, 0.0714648107], "rq": [-0.2988493588, -0.1527177234, -0.0845469705, 0.0850489178, 0.4736902308]}, "B": {"coef": [-0.9491643351, 0.1915643118, -0.944895985, 0.0820059133], "rq": [-0.2952697751, -0.1779703814, -0.0845419461, 0.0906310147, 0.5264408038]}, "C": {"coef": [-0.2067569834, 0.0134476463, 0.0479316849], "rq": [-0.2611022188, -0.1360035264, -0.0549608967, 0.1014047761, 0.4024326158]}},
     3: {"A": {"coef": [0.4080826739, -0.0424937531, 0.2798851455], "rq": [-0.8431048195, -0.2348911124, -0.0486852221, 0.210197539, 0.5500648005]}, "B": {"coef": [-1.7933742375, 0.4491856024, -2.5547016262, 0.3083721779], "rq": [-0.6797321984, -0.310746108, -0.096838751, 0.1932414709, 0.7101701475]}, "C": {"coef": [-0.168546047, -0.0009120029, 0.1549898214], "rq": [-0.5287427726, -0.1803460543, 0.0080658279, 0.239909578, 0.4769196734]}},
     4: {"A": {"coef": [0.8441522563, -0.0882548899, 0.2800776102], "rq": [-0.6445205421, -0.2768984637, -0.0510472439, 0.1737084794, 0.6065021623]}, "B": {"coef": [-1.5280484259, 0.4384044349, -2.720007845, 0.3102650846], "rq": [-0.6617211635, -0.3586388925, -0.0877187591, 0.1306284798, 0.7495944478]}, "C": {"coef": [0.0651092671, -0.0298309553, 0.1400454119], "rq": [-0.4823659067, -0.2669467178, 0.0020705968, 0.2373235592, 0.4838997242]}},
     5: {"A": {"coef": [0.8967786363, -0.0873701855, 0.3305840181], "rq": [-0.9704542198, -0.2786106427, -0.0493242783, 0.248393063, 0.6111482229]}, "B": {"coef": [-1.8328165977, 0.524135848, -3.1800148802, 0.3678258922], "rq": [-1.109051757, -0.3489324835, 0.0279857542, 0.2083017808, 0.63690419]}, "C": {"coef": [-0.0065787511, -0.0185570077, 0.1887192946], "rq": [-0.7688948738, -0.1903532749, -0.0131373278, 0.26565332, 0.5388962525]}},
     6: {"A": {"coef": [1.0513013856, -0.0948252363, 0.3450678477], "rq": [-0.9946562291, -0.4452278214, -0.0712817295, 0.284101978, 1.0300376166]}, "B": {"coef": [-2.0540951793, 0.6068936254, -3.6715194653, 0.3904776055], "rq": [-0.8727189403, -0.4832833202, -0.0125984315, 0.2337414819, 1.2031729497]}, "C": {"coef": [0.4875551726, -0.0532679288, 0.2375305691], "rq": [-0.8763428742, -0.3160683839, 0.0310880408, 0.2850655119, 0.9459291736]}},
     7: {"A": {"coef": [1.8981237792, -0.1685970583, 0.3607642153], "rq": [-0.8818728397, -0.3624076236, -0.1618855568, 0.2845464426, 1.0959624225]}, "B": {"coef": [-1.4567602445, 0.5847164048, -3.9165077699, 0.4090989003], "rq": [-0.8508971431, -0.4891741747, 0.0444575784, 0.2623112991, 1.0247266133]}, "C": {"coef": [1.2610500975, -0.1201114686, 0.2415625167], "rq": [-0.7306789141, -0.2958894999, -0.0925483744, 0.3046448385, 0.8747500011]}},
     8: {"A": {"coef": [2.6689754721, -0.234365869, 0.3477165118], "rq": [-0.812031205, -0.3692102556, -0.0766832345, 0.2304822023, 1.0832806925]}, "B": {"coef": [-0.9053042131, 0.5639907837, -4.1289437952, 0.3984750237], "rq": [-0.8960818604, -0.4439948868, -0.0217414045, 0.3020683606, 0.9877873799]}, "C": {"coef": [2.0646703728, -0.1859784529, 0.2409347642], "rq": [-0.7367440085, -0.4040930061, -0.1246374842, 0.2790058141, 0.9186417297]}},
     9: {"A": {"coef": [4.4984916266, -0.3938267995, 0.4155897605], "rq": [-0.9322346947, -0.4304628408, -0.043829073, 0.3203328884, 1.109685071]}, "B": {"coef": [2.966336862, -0.0720037474, -1.5553523723, 0.4345620284], "rq": [-1.0380748564, -0.516439857, 0.0151862897, 0.2675284373, 1.0427969242]}, "C": {"coef": [3.8409585894, -0.3363851039, 0.333715236], "rq": [-0.8205923969, -0.4752641907, 0.0317746819, 0.289683994, 0.9831630469]}},
    10: {"A": {"coef": [6.8939714254, -0.5975736043, 0.5595194197], "rq": [-1.2497473734, -0.5015652177, -0.069561603, 0.3617501528, 1.420622131]}, "B": {"coef": [8.5248028522, -1.0109850406, 2.4005304059, 0.5295892944], "rq": [-1.3378423621, -0.471887462, -0.0559586411, 0.2665605738, 1.420229584]}, "C": {"coef": [6.2261319697, -0.5309464094, 0.5314579956], "rq": [-1.0266055198, -0.532606819, -0.0347834384, 0.3745948239, 1.298398046]}},
    11: {"A": {"coef": [7.5081212537, -0.6477021727, 0.5817727824], "rq": [-1.4031571608, -0.4432352847, -0.0799336728, 0.443879311, 1.349316515]}, "B": {"coef": [9.4653829168, -1.1402446442, 2.8409183991, 0.5458738809], "rq": [-1.4689304918, -0.4584790147, -0.0345114838, 0.3426428829, 1.475064232]}, "C": {"coef": [6.516789291, -0.5531162421, 0.5424073526], "rq": [-1.2458439996, -0.5002867218, -0.0030579153, 0.3856049499, 1.2909516707]}},
    12: {"A": {"coef": [8.0620634066, -0.6897087176, 0.5984259092], "rq": [-1.4294385864, -0.3912433283, -0.0713991958, 0.4638252124, 1.300391463]}, "B": {"coef": [10.3356900916, -1.2577062642, 3.2560453696, 0.5572138259], "rq": [-1.4560743508, -0.3992063319, -0.0392155092, 0.3320516449, 1.3617908838]}, "C": {"coef": [7.049215787, -0.5924306785, 0.5585754897], "rq": [-1.2673458267, -0.4983587, -0.0447762933, 0.4430252642, 1.2565337552]}},
    13: {"A": {"coef": [9.1272432266, -0.7692500442, 0.6145917397], "rq": [-1.1372092735, -0.3032145283, -0.0813846958, 0.4359624872, 1.0734375385]}, "B": {"coef": [11.95544329, -1.4699320408, 3.9881027185, 0.5640074667], "rq": [-1.364820343, -0.4388725837, 0.0268260432, 0.2901168505, 1.3001423103]}, "C": {"coef": [8.3663106481, -0.6906434863, 0.58823287], "rq": [-1.1665303404, -0.4836237561, 0.0124350441, 0.5693436462, 1.1344812255]}},
    14: {"A": {"coef": [10.3139849075, -0.8598094397, 0.6668431729], "rq": [-1.2103410801, -0.3064305209, 0.1082242299, 0.5000082789, 0.9947918566]}, "B": {"coef": [14.8194869328, -1.9488026501, 6.0698976688, 0.5906819987], "rq": [-1.0687090083, -0.3670343451, 0.0785495363, 0.3910943738, 1.149771923]}, "C": {"coef": [9.0622625591, -0.7386874702, 0.5827252335], "rq": [-1.2362772084, -0.455200093, 0.0278594338, 0.6421687302, 1.0380727478]}},
    15: {"A": {"coef": [11.4498846444, -0.9479888483, 0.6760442661], "rq": [-1.2610481917, -0.2700527758, 0.0482336081, 0.5235694574, 0.9545531737]}, "B": {"coef": [17.0556824119, -2.2938033639, 7.457389449, 0.5827835516], "rq": [-1.3178197945, -0.2800123003, 0.0474556481, 0.4977435978, 0.9837519594]}, "C": {"coef": [9.5867330695, -0.7742975488, 0.5256104291], "rq": [-1.3129739662, -0.4573234643, -0.1370505274, 0.6993900191, 1.0190281841]}},
    16: {"A": {"coef": [12.2619529685, -1.01565685, 0.5362304631], "rq": [-1.1618452058, -0.3362542706, -0.0432345506, 0.5443226475, 0.932155988]}, "B": {"coef": [18.5182092294, -2.5167450603, 8.3098993269, 0.4314689968], "rq": [-0.9420877813, -0.3558846037, -0.1272876627, 0.4470359297, 1.238166453]}, "C": {"coef": [10.4288142365, -0.8426833238, 0.3958551984], "rq": [-1.1513652785, -0.4972536726, -0.1559632165, 0.6373263678, 1.1587590599]}},
    17: {"A": {"coef": [12.5816603919, -1.0378207997, 0.4803353362], "rq": [-1.2022716565, -0.4329831922, 0.0705777, 0.5988466715, 1.0872309015]}, "B": {"coef": [19.63902376, -2.7228312621, 9.2881126594, 0.363597602], "rq": [-1.1068100295, -0.3134362686, -0.1085837134, 0.4574992316, 1.3496968515]}, "C": {"coef": [10.6164426944, -0.8529557378, 0.3537660808], "rq": [-1.2118369194, -0.6592429996, 3.48964e-05, 0.6088424386, 1.1995925795]}},
    18: {"A": {"coef": [12.796779463, -1.0539977784, 0.3768858683], "rq": [-1.0831973493, -0.5805997797, -0.0500693119, 0.5984398883, 1.0111490184]}, "B": {"coef": [19.7949933201, -2.7264178645, 9.2267148335, 0.2609942266], "rq": [-0.9923226246, -0.4685387733, -0.0498475513, 0.4841382284, 1.3251099455]}, "C": {"coef": [10.7717460741, -0.8640296598, 0.2540971572], "rq": [-1.0850558994, -0.6766610556, -0.1854460398, 0.5635507323, 1.1275040844]}},
    19: {"A": {"coef": [12.6387083061, -1.040946875, 0.1855970018], "rq": [-1.0494174746, -0.5179649606, -0.058490249, 0.6166227206, 0.9060612682]}, "B": {"coef": [19.4828018273, -2.6752847879, 9.0136736071, 0.0731895065], "rq": [-1.0430237099, -0.4499197892, -0.0211264064, 0.3306738771, 1.285884748]}, "C": {"coef": [10.7037713126, -0.8583357417, 0.0754824732], "rq": [-1.0495135964, -0.5586505557, -0.2909494771, 0.5751527999, 1.1248213065]}},
    20: {"A": {"coef": [12.6736630485, -1.0427945228, 0.1665065589], "rq": [-0.9811306702, -0.3708374162, 0.0092032108, 0.587173867, 0.8419153078]}, "B": {"coef": [19.1941435491, -2.6019337235, 8.6118277397, 0.0597029019], "rq": [-0.81374986, -0.3967486757, 0.0200233581, 0.3520045966, 1.2177585514]}, "C": {"coef": [10.9247593409, -0.876066411, 0.0720574475], "rq": [-1.1120337919, -0.4973010499, 0.0085605993, 0.5766710396, 1.0515260721]}},
    21: {"A": {"coef": [12.8804426157, -1.0563365958, 0.2377996226], "rq": [-1.0418901863, -0.3585134001, 0.020422079, 0.5436859164, 0.7975460716]}, "B": {"coef": [18.9768328438, -2.5203782584, 8.116964112, 0.1368282333], "rq": [-0.9312443459, -0.4364270504, -0.0223308706, 0.4000217792, 1.0715257543]}, "C": {"coef": [11.1744290451, -0.8928226461, 0.1467453423], "rq": [-1.1953684848, -0.5332955556, 0.0082199306, 0.6643678369, 0.9986459708]}},
    22: {"A": {"coef": [12.7958597266, -1.0508377994, 0.2534747646], "rq": [-0.9876119071, -0.3516542478, -0.0386476003, 0.5043235068, 0.7913887728]}, "B": {"coef": [18.7497780831, -2.4810662478, 7.9330704563, 0.155171085], "rq": [-0.9489257237, -0.4453394134, -0.0202474069, 0.3910003697, 1.0090940271]}, "C": {"coef": [11.1530636903, -0.8930430994, 0.1666303029], "rq": [-1.1342289984, -0.563560138, -0.1681471378, 0.7276842144, 0.9200050916]}},
    23: {"A": {"coef": [12.4943614644, -1.0301490049, 0.2488615656], "rq": [-1.1175467576, -0.2895129568, -0.0146183242, 0.5161776831, 0.8113598919]}, "B": {"coef": [18.5057779153, -2.4702360139, 7.9711631907, 0.1508517594], "rq": [-0.9837594657, -0.4421391454, 0.0260469435, 0.3585007091, 0.9721817976]}, "C": {"coef": [10.9116144756, -0.8783432367, 0.165356699], "rq": [-1.2685715766, -0.4851007202, -0.1506733376, 0.7909984376, 0.8904022517]}},
    24: {"A": {"coef": [12.2087206674, -1.0161191008, 0.2292573364], "rq": [-1.0484147597, -0.3154854669, 0.0223447607, 0.5747595455, 0.8402416162]}, "B": {"coef": [18.5014252518, -2.5171380648, 8.2798799371, 0.1283689424], "rq": [-0.905422042, -0.4258545718, -0.0350540275, 0.4317917902, 0.9857249863]}, "C": {"coef": [10.6963001379, -0.8695568278, 0.1540741452], "rq": [-1.1572185565, -0.4594411954, -0.0984120882, 0.7748637771, 0.9094942635]}},
}


# The feature range those baked fits saw, per member, one [min, max] per feature: taken from the
# same 37 training rows in this session, so the fallback path clamps a target exactly as the
# runtime path does.
FALLBACK_RANGES = {
    "A": [[5.0998664278, 12.4292161968], [-5.4236430739, 1.6895423917]],
    "B": [[5.0998664278, 13.6821562925], [0.2600863758, 1.8720140081], [-5.4236430739, 1.6895423917]],
    "C": [[5.0998664278, 13.6821562925], [-5.4236430739, 1.6895423917]],
}


def fit_members(case):
    """Per-hour member fits plus the feature ranges they saw.

    Falls back to the baked coefficients (fitted once on the kit's comparables, whose ranges are
    the widest this build sees) when a case carries fewer than 10 usable comparables.
    """
    rows = training_rows(case)
    if len(rows) < 10:
        return FALLBACK_FITS, FALLBACK_RANGES
    fits = {}
    for h in HOURS:
        fits[h] = {m: fit_hour(rows, h, m) for m in ("A", "B", "C")}
    return fits, feature_ranges(rows)


def member_rows(fits, member, target_feats):
    """Per-hour quantile rows in log space for one member, with C's median smoothing."""
    row_log = {}
    medians = {}
    for h in HOURS:
        fit = fits[h][member]
        med = predict(fit["coef"], target_feats)
        medians[h] = med
        base = fit["rq"][2]
        row_log[h] = [med + (q - base) for q in fit["rq"]]
    if member == "C":
        smoothed = {}
        for h in HOURS:
            window = [h + d for d in (-1, 0, 1) if 2 <= h + d <= 24]
            smoothed[h] = sum(medians[w] for w in window) / len(window)
        for h in HOURS:
            shift = smoothed[h] - medians[h]
            row_log[h] = [q + shift for q in row_log[h]]
    return row_log


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    day = parse_ts(case["day_start"])
    history = [(parse_ts(p["from"]), p["viewers"]) for p in case["history_hourly"]]
    peak = pre_peak_viewers(history, day)
    if peak is None:
        peak = p1
    log_p1 = math.log(p1)
    f2 = log_p1 - math.log(max(1.0, float(peak)))
    fits, ranges = fit_members(case)
    target_feats = {
        m: clamp_features(features_for(m, log_p1, f2), ranges[m]) for m in ("A", "B", "C")
    }
    member_logs = {m: member_rows(fits, m, target_feats[m]) for m in ("A", "B", "C")}

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        q_log = [
            sum(member_logs[m][h][i] for m in ("A", "B", "C")) / 3.0
            for i in range(len(LEVELS))
        ]
        values = [p1 * math.exp(v) for v in q_log]
        out = []
        prev = 0.0
        for v in values:
            v = max(v, prev)
            v = min(max(v, 1e-6), 1e9)
            v = round2(v)
            if v < prev:
                v = prev
            out.append(v)
            prev = v
        hours.append({"hour": h, "q": out})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
