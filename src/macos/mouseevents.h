// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <ApplicationServices/ApplicationServices.h>
namespace eviacam {
// Creates Quartz events without posting them. The caller owns the returned
// reference and is responsible for the Accessibility/replay permission checks.
class MacMouseEvents {
public:
    CGEventRef Create(long x, long y, int flags, CGPoint current, double seconds);
private:
    bool leftDown = false, rightDown = false, middleDown = false;
    double lastLeftClick = -1;
    CGPoint lastClickPoint = CGPointZero;
    int clickCount = 1;
};
}
