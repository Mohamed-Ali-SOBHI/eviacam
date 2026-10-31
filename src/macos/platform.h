// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <string>
#include <vector>
namespace eviacam {
bool MacCameraPermission();
bool MacAccessibilityPermission(bool prompt);
std::vector<std::string> MacCameraNames();
void MacConfigureClickWindow(void* view, bool acceptsFocus);
}
