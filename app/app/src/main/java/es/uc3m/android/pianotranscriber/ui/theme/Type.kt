package es.uc3m.android.pianotranscriber.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.googlefonts.Font
import androidx.compose.ui.text.googlefonts.GoogleFont
import androidx.compose.ui.unit.sp
import es.uc3m.android.pianotranscriber.R

import androidx.compose.ui.text.font.Font

val AppFont = FontFamily(Font(R.font.dmsand_regular))

val AppTypography = Typography(
    bodyLarge   = TextStyle(fontFamily = AppFont, fontWeight = FontWeight.Normal, fontSize = 16.sp),
    bodyMedium  = TextStyle(fontFamily = AppFont, fontWeight = FontWeight.Normal, fontSize = 14.sp),
    bodySmall   = TextStyle(fontFamily = AppFont, fontWeight = FontWeight.Normal, fontSize = 12.sp),
    titleLarge  = TextStyle(fontFamily = AppFont, fontWeight = FontWeight.SemiBold, fontSize = 22.sp),
    titleMedium = TextStyle(fontFamily = AppFont, fontWeight = FontWeight.Medium, fontSize = 16.sp),
    labelSmall  = TextStyle(fontFamily = AppFont, fontWeight = FontWeight.Medium, fontSize = 11.sp),
)