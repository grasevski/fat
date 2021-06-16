//! Historical data loader.
use super::fxcm;
use arrayvec::ArrayString;
use chrono::{Datelike, Duration, NaiveDate};
use csv::{DeserializeRecordsIntoIter, Reader};
use enum_map::EnumMap;
use flate2::read::GzDecoder;
use num_traits::FromPrimitive;
use std::{convert::TryFrom, fmt::Write, io::Read};

/// Iterator to get historical data for all symbols.
pub struct History<R: Read, F: FnMut(&str) -> fxcm::Result<R>> {
    ix: i8,
    client: F,
    buf: EnumMap<fxcm::Symbol, Option<HistoryLoader<R>>>,
}

impl<R: Read, F: FnMut(&str) -> fxcm::Result<R>> History<R, F> {
    /// Initializes data loader for given date range.
    ///
    /// # Examples
    /// ```rust,editable
    /// let history = History::new(|_| [], Default::default(), None);
    /// assert_eq!(history.next(), None);
    /// ```
    pub fn new(client: F, begin: NaiveDate, end: Option<NaiveDate>) -> Self {
        let mut buf = EnumMap::default();
        for (k, v) in &mut buf {
            *v = Some(HistoryLoader::new(k, begin, end));
        }
        Self {
            ix: Default::default(),
            client,
            buf,
        }
    }
}

impl<R: Read, F: FnMut(&str) -> fxcm::Result<R>> Iterator for History<R, F> {
    type Item = fxcm::FallibleCandle;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ix) = FromPrimitive::from_i8(self.ix) {
            if let Some(ref mut loader) = self.buf[ix] {
                let ret = loader.next(&mut self.client);
                self.ix += 1;
                match i8::try_from(self.buf.len()) {
                    Ok(x) => {
                        self.ix %= x;
                        ret
                    }
                    Err(x) => Some(Err(fxcm::Error::from(x))),
                }
            } else {
                Some(Err(fxcm::Error::Initialization))
            }
        } else {
            Some(Err(fxcm::Error::IndexOutOfBounds(self.ix)))
        }
    }
}

struct HistoryLoader<R: Read> {
    symbol: fxcm::Symbol,
    current: NaiveDate,
    end: Option<NaiveDate>,
    rdr: Option<DeserializeRecordsIntoIter<GzDecoder<R>, fxcm::Candle>>,
}

impl<R: Read> HistoryLoader<R> {
    fn new(symbol: fxcm::Symbol, current: NaiveDate, end: Option<NaiveDate>) -> Self {
        Self {
            symbol,
            current,
            end,
            rdr: None,
        }
    }

    fn next(
        &mut self,
        mut client: impl FnMut(&str) -> fxcm::Result<R>,
    ) -> Option<fxcm::FallibleCandle> {
        if self.rdr.is_none() {
            const ENDPOINT: &str = "https://candledata.fxcorporate.com/m1";
            self.current -= Duration::days((self.current.ordinal0() % 7).into());
            if let Some(end) = self.end {
                if self.current >= end {
                    return None;
                }
            }
            let year = self.current.year();
            let week =
                1 + self.current.ordinal0() / 7 + u32::from(self.current.ordinal0() % 7 != 0);
            let mut url = ArrayString::<64>::new();
            if let Err(err) = write!(
                &mut url,
                "{}/{:?}/{}/{}.csv.gz",
                ENDPOINT, self.symbol, year, week
            ) {
                return Some(Err(fxcm::Error::from(err)));
            }
            self.current += Duration::weeks(1);
            match client(url.as_str()) {
                Ok(res) => {
                    let rdr = Reader::from_reader(GzDecoder::new(res));
                    self.rdr = Some(rdr.into_deserialize());
                }
                Err(res) => {
                    return Some(Err(res));
                }
            }
        }
        if let Some(rdr) = &mut self.rdr {
            if let Some(candle) = rdr.next() {
                match candle {
                    Ok(mut ret) => {
                        let t = ret.ts.naive_utc().date();
                        if t >= self.current {
                            ret.symbol = self.symbol;
                            if let Some(end) = self.end {
                                if t < end {
                                    Some(Ok(ret))
                                } else {
                                    None
                                }
                            } else {
                                Some(Ok(ret))
                            }
                        } else {
                            None
                        }
                    }
                    Err(ret) => Some(Err(fxcm::Error::from(ret))),
                }
            } else {
                self.next(client)
            }
        } else {
            None
        }
    }
}
