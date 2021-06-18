//! Historical data loader.
use super::fxcm;
use arrayvec::ArrayString;
use chrono::{Datelike, Duration, NaiveDate};
use csv::{DeserializeRecordsIntoIter, Reader};
use enum_map::EnumMap;
use flate2::read::GzDecoder;
use std::{fmt::Write, io::Read};

/// Iterator to get historical data for all symbols.
pub struct History<R: Read, F: FnMut(&str) -> fxcm::Result<R>> {
    begin: NaiveDate,
    end: Option<NaiveDate>,
    client: F,
    buf: EnumMap<fxcm::Symbol, (Option<fxcm::Candle>, Option<HistoryLoader<R>>)>,
}

impl<R: Read, F: FnMut(&str) -> fxcm::Result<R>> History<R, F> {
    /// Initializes data loader for given date range.
    pub fn new(mut client: F, begin: NaiveDate, end: Option<NaiveDate>) -> fxcm::Result<Self> {
        let mut buf = EnumMap::default();
        for (k, (candle, loader)) in &mut buf {
            let mut l = HistoryLoader::new(begin);
            if let Some(c) = l.next(k, begin, end, &mut client) {
                *candle = Some(c?);
                *loader = Some(l);
            } else {
                *loader = None;
            }
        }
        Ok(Self {
            begin,
            end,
            client,
            buf,
        })
    }
}

impl<R: Read, F: FnMut(&str) -> fxcm::Result<R>> Iterator for History<R, F> {
    type Item = fxcm::FallibleCandle;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(candle) = self.buf.values().flat_map(|x| x.0).min() {
            let (ref mut c, ref mut l) = self.buf[candle.symbol];
            if let Some(loader) = l {
                if let Some(x) = loader.next(candle.symbol, self.begin, self.end, &mut self.client)
                {
                    match x {
                        Ok(x) => {
                            *c = Some(x);
                        }
                        Err(err) => {
                            return Some(Err(err));
                        }
                    }
                } else {
                    *c = None;
                    *l = None;
                }
            } else {
                *c = None;
            }
            Some(Ok(candle))
        } else {
            None
        }
    }
}

struct HistoryLoader<R: Read> {
    current: NaiveDate,
    empty: bool,
    rdr: Option<DeserializeRecordsIntoIter<GzDecoder<R>, fxcm::Historical>>,
}

impl<R: Read> HistoryLoader<R> {
    fn new(current: NaiveDate) -> Self {
        Self {
            current,
            empty: true,
            rdr: None,
        }
    }

    fn next(
        &mut self,
        symbol: fxcm::Symbol,
        begin: NaiveDate,
        end: Option<NaiveDate>,
        mut client: impl FnMut(&str) -> fxcm::Result<R>,
    ) -> Option<fxcm::FallibleCandle> {
        if self.rdr.is_none() {
            const ENDPOINT: &str = "https://candledata.fxcorporate.com/m1";
            self.current -= Duration::days((self.current.ordinal0() % 7).into());
            if let Some(end) = end {
                if self.current >= end {
                    return None;
                }
            }
            let year = self.current.year();
            let week =
                1 + self.current.ordinal0() / 7 + u32::from(self.current.ordinal0() % 7 != 0);
            let mut url = ArrayString::<64>::new();
            if let Err(err) = write!(&mut url, "{}/{}/{}/{}.csv.gz", ENDPOINT, symbol, year, week) {
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
            while let Some(candle) = rdr.next() {
                self.empty = false;
                match candle {
                    Ok(candle) => match fxcm::Candle::new(symbol, candle) {
                        Ok(candle) => {
                            let t = candle.ts.naive_utc().date();
                            if t >= begin {
                                return if let Some(end) = end {
                                    if t < end {
                                        Some(Ok(candle))
                                    } else {
                                        None
                                    }
                                } else {
                                    Some(Ok(candle))
                                };
                            }
                        }
                        Err(err) => {
                            return Some(Err(err));
                        }
                    },
                    Err(ret) => {
                        return Some(Err(fxcm::Error::from(ret)));
                    }
                }
            }
            if self.empty {
                None
            } else {
                self.rdr = None;
                self.empty = true;
                self.next(symbol, begin, end, client)
            }
        } else {
            None
        }
    }
}
